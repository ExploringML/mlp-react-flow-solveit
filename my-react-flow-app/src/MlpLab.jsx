import { useCallback, useLayoutEffect, useMemo, useRef, useState } from 'react'
import {
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  MiniMap,
  Position,
  ReactFlow,
  useEdgesState,
  useNodesState,
} from '@xyflow/react'
import {
  EXAMPLES,
  cloneNetwork,
  createNetwork,
  evaluate,
  forward,
  mulberry32,
  parameterCount,
  trainEpoch,
} from './mlp.js'
import './lab.css'

const COLORS = ['#59d5ff', '#8f7cff', '#ec5faf', '#ff9a3c', '#35d39a', '#ffd24a']
const wait = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds))
const formatLoss = (value) => Number.isFinite(value) ? value.toFixed(value < 0.01 ? 4 : 3) : '—'
const resizeInputs = (values, count) => Array.from({ length: count }, (_, index) => values[index] ?? 0)
const resizeRows = (rows, count) => rows.map((row) => ({ ...row, x: resizeInputs(row.x, count) }))

function LayerGroup({ data }) {
  return (
    <div className={`layer-group layer-group-${data.kind}`}>
      <div className="layer-kicker">{data.kicker}</div>
      <div className="layer-title">{data.title}</div>
      <div className="layer-subtitle">{data.subtitle}</div>
    </div>
  )
}

function InputNode({ data }) {
  return (
    <div className="network-node input-node">
      <div className="node-topline">
        <span className="node-symbol">x<sub>{data.index + 1}</sub></span>
        <span className="node-value">{data.value.toFixed(2)}</span>
      </div>
      <div className="node-label" title={data.label}>{data.label}</div>
      <input
        aria-label={data.label}
        className="nodrag nopan input-slider"
        min="0"
        max="1"
        step="0.01"
        type="range"
        value={data.value}
        onChange={(event) => data.onChange(data.index, Number(event.target.value))}
      />
      <Handle type="source" position={Position.Right} />
    </div>
  )
}

function HiddenNode({ data }) {
  const intensity = Math.min(1, Math.abs(data.value))
  return (
    <div
      className="network-node hidden-node"
      style={{ '--node-color': data.color, '--activation': intensity }}
      title={`Activation: ${data.value.toFixed(4)}`}
    >
      <Handle type="target" position={Position.Left} />
      <div className="hidden-orb">h<sub>{data.layer}{data.index + 1}</sub></div>
      <div className="hidden-value">{data.value.toFixed(2)}</div>
      <Handle type="source" position={Position.Right} />
    </div>
  )
}

function OutputNode({ data }) {
  return (
    <div className="network-node output-node" style={{ '--node-color': data.color }}>
      <Handle type="target" position={Position.Left} />
      <div className="output-symbol">ŷ<sub>{data.index + 1}</sub></div>
      <div className="output-name">{data.label}</div>
      <div className="output-value">{data.display}</div>
      {data.probability != null && (
        <div className="probability-track">
          <span style={{ width: `${Math.max(1, data.probability * 100)}%` }} />
        </div>
      )}
    </div>
  )
}

const nodeTypes = {
  layerGroup: LayerGroup,
  input: InputNode,
  hidden: HiddenNode,
  output: OutputNode,
}

function edgeAppearance(weight) {
  const magnitude = Math.abs(weight)
  return {
    stroke: weight >= 0 ? '#20c7a0' : '#ef5b88',
    strokeWidth: 0.65 + Math.min(4.2, magnitude * 2.6),
    opacity: 0.15 + Math.min(0.72, magnitude * 0.46),
  }
}

function buildGraph(config, network, inputs, prediction, onInputChange, training = false) {
  const layerSizes = network.sizes
  const maxUnits = Math.max(...layerSizes)
  const rowHeight = 67
  const groupHeight = Math.max(650, 96 + maxUnits * rowHeight)
  const widths = layerSizes.map((_, index) => {
    if (index === 0) return 245
    if (index === layerSizes.length - 1) return 260
    return 205
  })
  const columns = []
  let cursor = 0
  widths.forEach((width) => {
    columns.push(cursor)
    cursor += width + 115
  })

  const nodes = []
  const edges = []
  layerSizes.forEach((size, layerIndex) => {
    const isInput = layerIndex === 0
    const isOutput = layerIndex === layerSizes.length - 1
    const kind = isInput ? 'input' : isOutput ? 'output' : 'hidden'
    const title = isInput ? 'Observed features' : isOutput ? 'Prediction' : `Hidden layer ${layerIndex}`
    const subtitle = isInput
      ? `${size} normalized values`
      : isOutput
        ? (config.task === 'classification' ? `${size} class probabilities` : 'continuous estimate')
        : `${size} ${network.hiddenActivation.toUpperCase()} units`

    nodes.push({
      id: `group-${layerIndex}`,
      type: 'layerGroup',
      position: { x: columns[layerIndex], y: 0 },
      data: { kicker: `Network layer ${layerIndex + 1}`, title, subtitle, kind },
      draggable: false,
      selectable: false,
      connectable: false,
      style: { width: widths[layerIndex], height: groupHeight },
      zIndex: -2,
    })

    const nodeHeight = isInput ? 55 : isOutput ? 94 : 52
    const totalHeight = size * rowHeight
    const start = 85 + Math.max(0, (groupHeight - 100 - totalHeight) / 2)
    for (let unit = 0; unit < size; unit += 1) {
      const id = `${kind}-${layerIndex}-${unit}`
      let data
      if (isInput) {
        data = {
          index: unit,
          label: config.featureNames[unit] || `Input ${unit + 1}`,
          value: inputs[unit],
          onChange: onInputChange,
        }
      } else if (isOutput) {
        const value = prediction[unit]
        data = {
          index: unit,
          label: config.outputNames[unit],
          color: COLORS[unit % COLORS.length],
          probability: config.task === 'classification' ? value : null,
          display: config.task === 'classification'
            ? `${(value * 100).toFixed(1)}%`
            : `$${Math.max(0, value * 1200).toFixed(0)}k`,
        }
      } else {
        const activation = forward(network, inputs).activations[layerIndex][unit]
        data = {
          layer: layerIndex,
          index: unit,
          value: activation,
          color: COLORS[(unit + layerIndex) % COLORS.length],
        }
      }

      nodes.push({
        id,
        type: kind,
        parentId: `group-${layerIndex}`,
        extent: 'parent',
        position: {
          x: isInput ? 20 : isOutput ? 30 : 32,
          y: start + unit * rowHeight,
        },
        data,
        style: {
          width: isInput ? 205 : isOutput ? 200 : 140,
          height: nodeHeight,
        },
        zIndex: 2,
      })
    }
  })

  let edgeIndex = 0
  network.weights.forEach((matrix, matrixIndex) => {
    const sourceKind = matrixIndex === 0 ? 'input' : 'hidden'
    const targetLayer = matrixIndex + 1
    const targetKind = targetLayer === network.sizes.length - 1 ? 'output' : 'hidden'
    matrix.forEach((row, target) => {
      row.forEach((weight, source) => {
        const appearance = edgeAppearance(weight)
        edges.push({
          id: `w-${matrixIndex}-${source}-${target}`,
          source: `${sourceKind}-${matrixIndex}-${source}`,
          target: `${targetKind}-${targetLayer}-${target}`,
          type: 'default',
          className: training ? 'calculation-edge' : '',
          style: {
            ...appearance,
            '--flash-delay': `${(edgeIndex % 48) * 22}ms`,
            '--flash-color': appearance.stroke,
          },
          interactionWidth: 8,
          data: { weight, layer: matrixIndex, source, target },
          zIndex: 0,
        })
        edgeIndex += 1
      })
    })
  })

  return { nodes, edges, width: cursor, height: groupHeight }
}

function mergeGraphNodes(current, next) {
  const previous = new Map(current.map((node) => [node.id, node]))
  return next.map((node) => {
    const old = previous.get(node.id)
    if (!old) return node
    return {
      ...old,
      ...node,
      data: node.data,
      style: node.style,
      measured: old.measured,
      selected: old.selected,
    }
  })
}

function mergeGraphEdges(current, next) {
  const previous = new Map(current.map((edge) => [edge.id, edge]))
  return next.map((edge) => ({
    ...previous.get(edge.id),
    ...edge,
    data: edge.data,
    style: edge.style,
  }))
}

function NetworkFlow({ graph, onSelect }) {
  const [nodes, setNodes, onNodesChange] = useNodesState(graph.nodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(graph.edges)

  useLayoutEffect(() => {
    setNodes((current) => mergeGraphNodes(current, graph.nodes))
    setEdges((current) => mergeGraphEdges(current, graph.edges))
  }, [graph, setEdges, setNodes])

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onNodeClick={(_, node) => node.type !== 'layerGroup' && onSelect({ type: 'node', node })}
      onEdgeClick={(_, edge) => onSelect({ type: 'edge', data: edge.data })}
      fitView
      fitViewOptions={{ padding: 0.06, minZoom: 0.22, maxZoom: 1, duration: 220 }}
      minZoom={0.12}
      maxZoom={1.6}
      nodesDraggable={false}
      colorMode="dark"
      proOptions={{ hideAttribution: true }}
    >
      <Controls showInteractive={false} />
      <MiniMap
        pannable
        zoomable
        nodeColor={(node) => node.type === 'input' ? '#49b5d8' : node.type === 'output' ? '#d8559d' : '#6e5fd7'}
        maskColor="rgba(5, 8, 16, .72)"
      />
      <Background variant={BackgroundVariant.Dots} gap={18} size={1} color="#24304a" />
    </ReactFlow>
  )
}

function MetricCard({ label, value, tone }) {
  return (
    <div className={`metric-card ${tone || ''}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function LearningCurve({ history, selected, onSelect, showAccuracy, task }) {
  const width = 920
  const height = 245
  const pad = { left: 50, right: 46, top: 20, bottom: 34 }
  const plotWidth = width - pad.left - pad.right
  const plotHeight = height - pad.top - pad.bottom
  const maximumEpoch = Math.max(1, history.at(-1)?.epoch || 1)
  const maxLoss = Math.max(0.05, ...history.flatMap((item) => [item.trainLoss, item.validationLoss])) * 1.08
  const x = (epoch) => pad.left + (epoch / maximumEpoch) * plotWidth
  const lossY = (value) => pad.top + plotHeight - (value / maxLoss) * plotHeight
  const accuracyY = (value) => pad.top + plotHeight - value * plotHeight
  const path = (key, scale) => history.map((item, index) =>
    `${index ? 'L' : 'M'} ${x(item.epoch).toFixed(2)} ${scale(item[key]).toFixed(2)}`,
  ).join(' ')
  const selectedItem = history.find((item) => item.epoch === selected) || history.at(-1)

  if (!history.length) {
    return <div className="empty-chart">Run training to plot training and validation performance.</div>
  }

  return (
    <div className="chart-shell">
      <svg className="learning-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Learning curve">
        {[0, 0.25, 0.5, 0.75, 1].map((fraction) => {
          const y = pad.top + plotHeight * fraction
          return <line key={fraction} x1={pad.left} x2={width - pad.right} y1={y} y2={y} className="chart-grid" />
        })}
        <path d={path('trainLoss', lossY)} className="curve curve-train" />
        <path d={path('validationLoss', lossY)} className="curve curve-validation" />
        {showAccuracy && task === 'classification' && (
          <path d={path('accuracy', accuracyY)} className="curve curve-accuracy" />
        )}
        {history.map((item) => (
          <circle
            key={item.epoch}
            cx={x(item.epoch)}
            cy={lossY(item.validationLoss)}
            r={item.epoch === selectedItem.epoch ? 6 : 3}
            className="chart-point"
            onClick={() => onSelect(item.epoch)}
          />
        ))}
        <line
          x1={x(selectedItem.epoch)} x2={x(selectedItem.epoch)}
          y1={pad.top} y2={pad.top + plotHeight}
          className="chart-cursor"
        />
        <text x={pad.left} y={height - 9} className="chart-axis">Start</text>
        <text x={width - pad.right} y={height - 9} textAnchor="end" className="chart-axis">Epoch {maximumEpoch}</text>
        <text x={8} y={pad.top + 4} className="chart-axis">{maxLoss.toFixed(2)}</text>
        <text x={8} y={pad.top + plotHeight} className="chart-axis">0</text>
        {showAccuracy && task === 'classification' && <text x={width - 34} y={pad.top + 4} className="chart-axis">100%</text>}
      </svg>
      <div className="chart-legend">
        <span><i className="legend-train" />Training loss</span>
        <span><i className="legend-validation" />Validation loss</span>
        {showAccuracy && task === 'classification' && <span><i className="legend-accuracy" />Validation accuracy</span>}
      </div>
    </div>
  )
}

function Inspector({ selected, network, config }) {
  if (!selected) {
    return (
      <div className="inspector-empty">
        Select a neuron or connection to inspect its current value or weight.
      </div>
    )
  }
  if (selected.type === 'edge') {
    return (
      <div className="inspector-content">
        <span className="eyebrow">Selected connection</span>
        <strong>Weight {selected.data.weight.toFixed(5)}</strong>
        <p>{selected.data.weight >= 0 ? 'Positive' : 'Negative'} influence from unit {selected.data.source + 1} to unit {selected.data.target + 1}.</p>
      </div>
    )
  }
  const node = selected.node
  if (node.type === 'input') {
    return (
      <div className="inspector-content">
        <span className="eyebrow">Observed feature</span>
        <strong>{node.data.label}</strong>
        <p>Normalized value: {node.data.value.toFixed(4)}. Drag its slider to recalculate the complete forward pass.</p>
      </div>
    )
  }
  if (node.type === 'hidden') {
    return (
      <div className="inspector-content">
        <span className="eyebrow">Hidden neuron</span>
        <strong>h<sub>{node.data.layer}{node.data.index + 1}</sub> · {network.hiddenActivation.toUpperCase()}</strong>
        <p>Current activation: {node.data.value.toFixed(5)}.</p>
      </div>
    )
  }
  if (node.type === 'output') {
    return (
      <div className="inspector-content">
        <span className="eyebrow">Model output</span>
        <strong>{node.data.label}</strong>
        <p>{config.task === 'classification' ? `Predicted probability: ${node.data.display}.` : `Predicted synthetic sale price: ${node.data.display}.`}</p>
      </div>
    )
  }
  return null
}

export default function MlpLab() {
  const initialConfig = EXAMPLES.mars
  const initialInputCount = initialConfig.featureNames.length
  const [exampleKey, setExampleKey] = useState('mars')
  const [inputCount, setInputCount] = useState(initialInputCount)
  const [hidden, setHidden] = useState([...initialConfig.hidden])
  const [activation, setActivation] = useState(initialConfig.activation)
  const [network, setNetwork] = useState(() => createNetwork(
    [initialInputCount, ...initialConfig.hidden, initialConfig.outputNames.length],
    initialConfig.activation,
    initialConfig.task,
    initialConfig.seed,
  ))
  const [inputs, setInputs] = useState([...initialConfig.dataset.validation[0].x])
  const [sampleIndex, setSampleIndex] = useState(0)
  const [epochs, setEpochs] = useState(initialConfig.epochs)
  const [batchSize, setBatchSize] = useState(initialConfig.batchSize)
  const [learningRate, setLearningRate] = useState(initialConfig.learningRate)
  const [trainingRows, setTrainingRows] = useState(initialConfig.dataset.train.length)
  const [history, setHistory] = useState([])
  const [selectedEpoch, setSelectedEpoch] = useState(0)
  const [showAccuracy, setShowAccuracy] = useState(true)
  const [training, setTraining] = useState(false)
  const [status, setStatus] = useState('Ready to train')
  const [selected, setSelected] = useState(null)
  const stopRef = useRef(false)
  const runTokenRef = useRef(0)
  const fileRef = useRef(null)
  const [flowVersion, setFlowVersion] = useState(0)
  const config = EXAMPLES[exampleKey]
  const dataset = useMemo(() => ({
    train: resizeRows(config.dataset.train, inputCount),
    validation: resizeRows(config.dataset.validation, inputCount),
  }), [config, inputCount])

  const output = useMemo(() => forward(network, inputs).output, [network, inputs])
  const validationTarget = dataset.validation[sampleIndex]?.y
  const predictedIndex = config.task === 'classification' ? output.indexOf(Math.max(...output)) : 0
  const actualIndex = config.task === 'classification' ? validationTarget.indexOf(Math.max(...validationTarget)) : 0

  const updateInput = useCallback((index, value) => {
    setInputs((current) => current.map((item, itemIndex) => itemIndex === index ? value : item))
  }, [])

  const graph = useMemo(
    () => buildGraph(config, network, inputs, output, updateInput, training),
    [config, network, inputs, output, updateInput, training],
  )


  const freshNetwork = useCallback((nextConfig = config, nextInputCount = inputCount, nextHidden = hidden, nextActivation = activation, offset = 0) =>
    createNetwork(
      [nextInputCount, ...nextHidden, nextConfig.outputNames.length],
      nextActivation,
      nextConfig.task,
      nextConfig.seed + offset,
    ), [activation, config, hidden, inputCount])

  function resetRunState(message = 'Model reset to example defaults') {
    setHistory([])
    setSelectedEpoch(0)
    setSelected(null)
    setStatus(message)
  }

  function resetModel() {
    runTokenRef.current += 1
    stopRef.current = true
    setTraining(false)
    const defaultInputCount = config.featureNames.length
    const defaultHidden = [...config.hidden]
    setInputCount(defaultInputCount)
    setHidden(defaultHidden)
    setActivation(config.activation)
    setEpochs(config.epochs)
    setBatchSize(config.batchSize)
    setLearningRate(config.learningRate)
    setTrainingRows(config.dataset.train.length)
    setSampleIndex(0)
    setInputs([...config.dataset.validation[0].x])
    setNetwork(createNetwork(
      [defaultInputCount, ...defaultHidden, config.outputNames.length],
      config.activation,
      config.task,
      config.seed,
    ))
    resetRunState()
    setFlowVersion((version) => version + 1)
  }

  function loadExample(key) {
    if (training) return
    const next = EXAMPLES[key]
    const nextInputCount = next.featureNames.length
    const nextHidden = [...next.hidden]
    setExampleKey(key)
    setInputCount(nextInputCount)
    setHidden(nextHidden)
    setActivation(next.activation)
    setEpochs(next.epochs)
    setBatchSize(next.batchSize)
    setLearningRate(next.learningRate)
    setTrainingRows(next.dataset.train.length)
    setSampleIndex(0)
    setInputs([...next.dataset.validation[0].x])
    setNetwork(createNetwork(
      [nextInputCount, ...nextHidden, next.outputNames.length],
      next.activation,
      next.task,
      next.seed,
    ))
    resetRunState(`${next.title} loaded`)
    setFlowVersion((version) => version + 1)
  }

  function applyArchitecture(nextInputCount, nextHidden, nextActivation = activation) {
    if (training) return
    const cleanInputCount = Math.max(1, Math.min(16, Number(nextInputCount) || 1))
    const cleanHidden = nextHidden.map((value) => Math.max(2, Math.min(24, Number(value) || 2)))
    setInputCount(cleanInputCount)
    setHidden(cleanHidden)
    setActivation(nextActivation)
    setInputs((current) => resizeInputs(current, cleanInputCount))
    setNetwork(freshNetwork(config, cleanInputCount, cleanHidden, nextActivation))
    resetRunState('Architecture changed; weights reinitialized')
    setFlowVersion((version) => version + 1)
  }

  function nextSample() {
    const next = (sampleIndex + 1) % dataset.validation.length
    setSampleIndex(next)
    setInputs([...dataset.validation[next].x])
    setSelected(null)
    setStatus(`Loaded validation example ${next + 1}`)
  }

  async function runTraining() {
    if (training) return
    const runToken = runTokenRef.current + 1
    runTokenRef.current = runToken
    stopRef.current = false
    setTraining(true)
    setSelected(null)
    setStatus('Tracing the forward and backward calculations…')
    const trainRows = dataset.train.slice(0, trainingRows)
    let current = cloneNetwork(network)
    let currentHistory = []
    const record = (epoch) => {
      const trainMetrics = evaluate(current, trainRows)
      const validationMetrics = evaluate(current, dataset.validation)
      return {
        epoch,
        trainLoss: trainMetrics.loss,
        validationLoss: validationMetrics.loss,
        accuracy: validationMetrics.accuracy,
        mae: validationMetrics.mae,
      }
    }
    currentHistory = [record(0)]
    setHistory(currentHistory)
    setSelectedEpoch(0)
    const rng = mulberry32(config.seed + history.length * 97 + 19)

    for (let epoch = 1; epoch <= epochs; epoch += 1) {
      trainEpoch(current, trainRows, learningRate, batchSize, rng)
      if (runToken !== runTokenRef.current) return
      const item = record(epoch)
      currentHistory = [...currentHistory, item]
      setHistory(currentHistory)
      setNetwork(cloneNetwork(current))
      setSelectedEpoch(epoch)
      setStatus(`Epoch ${epoch}/${epochs} · tracing weighted connections · validation loss ${formatLoss(item.validationLoss)}`)
      if (stopRef.current) break
      await wait(45)
    }

    if (runToken !== runTokenRef.current) return
    setNetwork(cloneNetwork(current))
    setTraining(false)
    const last = currentHistory.at(-1)
    setStatus(stopRef.current
      ? `Training stopped after epoch ${last.epoch}`
      : `Training complete · validation loss ${formatLoss(last.validationLoss)}`,
    )
  }

  function stopTraining() {
    stopRef.current = true
    setStatus('Stopping after the current epoch…')
  }

  function exportModel() {
    const payload = JSON.stringify({ version: 1, exampleKey, inputCount, hidden, activation, network }, null, 2)
    const url = URL.createObjectURL(new Blob([payload], { type: 'application/json' }))
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `${exampleKey}-mlp-model.json`
    anchor.click()
    URL.revokeObjectURL(url)
  }

  async function importModel(event) {
    const file = event.target.files?.[0]
    if (!file || training) return
    try {
      const payload = JSON.parse(await file.text())
      const nextConfig = EXAMPLES[payload.exampleKey]
      if (!nextConfig || !payload.network?.weights || !payload.network?.sizes) throw new Error('Unsupported model file')
      setExampleKey(payload.exampleKey)
      setHidden(payload.network.sizes.slice(1, -1))
      setActivation(payload.activation)
      setNetwork(payload.network)
      const nextInputCount = Math.max(1, Math.min(16, payload.network.sizes[0]))
      setInputCount(nextInputCount)
      setInputs(resizeInputs(nextConfig.dataset.validation[0].x, nextInputCount))
      setSampleIndex(0)
      setEpochs(nextConfig.epochs)
      setBatchSize(nextConfig.batchSize)
      setLearningRate(nextConfig.learningRate)
      setTrainingRows(nextConfig.dataset.train.length)
      setFlowVersion((version) => version + 1)
      resetRunState('Imported model parameters')
    } catch (error) {
      setStatus(`Import failed: ${error.message}`)
    } finally {
      event.target.value = ''
    }
  }

  const currentMetric = history.find((item) => item.epoch === selectedEpoch) || history.at(-1)
  const architecture = network.sizes.join(' → ')
  const sampleCorrect = config.task === 'classification' ? predictedIndex === actualIndex : null

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark">MLP</div>
          <div>
            <span className="eyebrow">Interactive neural-network laboratory</span>
            <h1>{config.title}</h1>
          </div>
        </div>
        <div className="toolbar">
          <label className="select-wrap">
            <span>Example</span>
            <select value={exampleKey} onChange={(event) => loadExample(event.target.value)} disabled={training}>
              {Object.entries(EXAMPLES).map(([key, item]) => <option key={key} value={key}>{item.short}</option>)}
            </select>
          </label>
          <button className="button primary" onClick={training ? stopTraining : runTraining}>
            {training ? 'Stop training' : 'Train model'}
          </button>
          <button className="button" onClick={resetModel}>Reset model</button>
          <button className="button" onClick={nextSample}>Next sample</button>
          <button className="button" onClick={exportModel}>Download</button>
          <button className="button" onClick={() => fileRef.current?.click()} disabled={training}>Import</button>
          <input ref={fileRef} hidden type="file" accept="application/json" onChange={importModel} />
        </div>
      </header>

      <section className="workspace">
        <div className="canvas-panel">
          <div className="canvas-heading">
            <div>
              <span className="eyebrow">Live forward pass</span>
              <h2>{architecture} · {parameterCount(network.sizes)} trainable parameters</h2>
            </div>
            <div className="prediction-summary">
              <span>{config.task === 'classification' ? 'Predicted class' : 'Estimated price'}</span>
              <strong style={{ color: COLORS[predictedIndex % COLORS.length] }}>
                {config.task === 'classification' ? config.outputNames[predictedIndex] : `$${Math.max(0, output[0] * 1200).toFixed(0)}k`}
              </strong>
            </div>
          </div>
          <div className="flow-stage">
            <NetworkFlow
              key={`${network.sizes.join('-')}-${flowVersion}`}
              graph={graph}
              onSelect={setSelected}
            />
          </div>
          <div className="statusbar">
            <span className={`status-dot ${training ? 'running' : ''}`} />
            <strong>{status}</strong>
            <span>Validation row {sampleIndex + 1}/{config.dataset.validation.length}</span>
            {config.task === 'classification' && (
              <span className={sampleCorrect ? 'success-text' : 'warning-text'}>
                Actual: {config.outputNames[actualIndex]}
              </span>
            )}
          </div>
        </div>

        <aside className="side-panel">
          <section className="control-card">
            <div className="section-title">
              <div><span className="eyebrow">Architecture</span><h3>Design the network</h3></div>
              <span className="pill">{parameterCount(network.sizes)} params</span>
            </div>
            <div className="layer-editor">
              <div className="layer-editor-row fixed-layer">
                <div><span className="layer-editor-kicker">Input layer</span><strong>Observed features</strong></div>
                <label><span>Nodes</span><input type="number" min="1" max="16" value={inputCount} disabled={training} onChange={(event) => applyArchitecture(Number(event.target.value), hidden)} /></label>
                <span className="fixed-badge">Required</span>
              </div>
              {hidden.map((units, index) => (
                <div className="layer-editor-row" key={index}>
                  <div><span className="layer-editor-kicker">Hidden layer {index + 1}</span><strong>{units} neurons</strong></div>
                  <label><span>Nodes</span><input type="number" min="2" max="24" value={units} disabled={training} onChange={(event) => applyArchitecture(inputCount, hidden.map((item, itemIndex) => itemIndex === index ? Number(event.target.value) : item))} /></label>
                  <button className="icon-button danger" aria-label={`Delete hidden layer ${index + 1}`} title="Delete hidden layer" disabled={training} onClick={() => applyArchitecture(inputCount, hidden.filter((_, itemIndex) => itemIndex !== index))}>×</button>
                </div>
              ))}
              <div className="layer-editor-row fixed-layer">
                <div><span className="layer-editor-kicker">Output layer</span><strong>{config.task === 'classification' ? 'Class probabilities' : 'Prediction'}</strong></div>
                <label><span>Nodes</span><input type="number" value={config.outputNames.length} disabled /></label>
                <span className="fixed-badge">Required</span>
              </div>
            </div>
            <button className="button add-layer-button" disabled={training || hidden.length >= 4} onClick={() => applyArchitecture(inputCount, [...hidden, 8])}>+ Add hidden layer</button>
            <label className="field-control">
              <span>Hidden activation</span>
              <select value={activation} disabled={training || hidden.length === 0} onChange={(event) => applyArchitecture(inputCount, hidden, event.target.value)}>
                <option value="relu">ReLU</option>
                <option value="tanh">Tanh</option>
                <option value="sigmoid">Sigmoid</option>
              </select>
            </label>
          </section>

          <section className="control-card">
            <div className="section-title"><div><span className="eyebrow">Optimizer</span><h3>Training controls</h3></div></div>
            <div className="field-grid">
              <label className="field-control"><span>Epochs</span><input type="number" min="1" max="300" value={epochs} disabled={training} onChange={(event) => setEpochs(Number(event.target.value))} /></label>
              <label className="field-control"><span>Batch size</span><select value={batchSize} disabled={training} onChange={(event) => setBatchSize(Number(event.target.value))}><option>8</option><option>16</option><option>32</option><option>64</option></select></label>
              <label className="field-control"><span>Learning rate</span><input type="number" min="0.001" max="0.2" step="0.005" value={learningRate} disabled={training} onChange={(event) => setLearningRate(Number(event.target.value))} /></label>
              <label className="field-control"><span>Training rows</span><input type="number" min="32" max={config.dataset.train.length} step="16" value={trainingRows} disabled={training} onChange={(event) => setTrainingRows(Number(event.target.value))} /></label>
            </div>
          </section>

          <section className="control-card inspector-card">
            <div className="section-title"><div><span className="eyebrow">Inspector</span><h3>Neuron or connection</h3></div></div>
            <Inspector selected={selected} network={network} config={config} />
          </section>
        </aside>
      </section>

      <section className="results-panel">
        <div className="results-heading">
          <div>
            <span className="eyebrow">Learning curves</span>
            <h2>Training and validation performance</h2>
          </div>
          {config.task === 'classification' && (
            <label className="checkbox-control">
              <input type="checkbox" checked={showAccuracy} onChange={(event) => setShowAccuracy(event.target.checked)} />
              Validation accuracy
            </label>
          )}
        </div>
        <LearningCurve
          history={history}
          selected={selectedEpoch}
          onSelect={setSelectedEpoch}
          showAccuracy={showAccuracy}
          task={config.task}
        />
        <div className="metric-row">
          <MetricCard label={currentMetric ? `Training loss · ${currentMetric.epoch === 0 ? 'Start' : `epoch ${currentMetric.epoch}`}` : 'Training loss'} value={currentMetric ? formatLoss(currentMetric.trainLoss) : '—'} tone="cyan" />
          <MetricCard label="Validation loss" value={currentMetric ? formatLoss(currentMetric.validationLoss) : '—'} tone="violet" />
          {config.task === 'classification'
            ? <MetricCard label="Validation accuracy" value={currentMetric ? `${(currentMetric.accuracy * 100).toFixed(1)}%` : '—'} tone="green" />
            : <MetricCard label="Validation MAE" value={currentMetric ? `$${(currentMetric.mae * 1200).toFixed(1)}k` : '—'} tone="green" />}
          <MetricCard label="Dataset split" value={`${trainingRows} / ${config.dataset.validation.length}`} tone="amber" />
        </div>
      </section>
    </main>
  )
}
