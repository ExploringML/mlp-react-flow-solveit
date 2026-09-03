const clamp = (value, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, value))

export function mulberry32(seed) {
  let state = seed >>> 0
  return () => {
    state += 0x6d2b79f5
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function normal(rng) {
  const u = Math.max(rng(), 1e-9)
  const v = rng()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

const sigmoid = (x) => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, x))))

function activate(value, name) {
  if (name === 'relu') return Math.max(0, value)
  if (name === 'sigmoid') return sigmoid(value)
  if (name === 'tanh') return Math.tanh(value)
  return value
}

function activationDerivative(z, a, name) {
  if (name === 'relu') return z > 0 ? 1 : 0
  if (name === 'sigmoid') return a * (1 - a)
  if (name === 'tanh') return 1 - a * a
  return 1
}

function softmax(values) {
  const maximum = Math.max(...values)
  const exps = values.map((value) => Math.exp(value - maximum))
  const total = exps.reduce((sum, value) => sum + value, 0)
  return exps.map((value) => value / total)
}

export function createNetwork(sizes, hiddenActivation = 'relu', task = 'classification', seed = 1) {
  const rng = mulberry32(seed)
  const weights = []
  const biases = []
  for (let layer = 1; layer < sizes.length; layer += 1) {
    const fanIn = sizes[layer - 1]
    const scale = hiddenActivation === 'relu' ? Math.sqrt(2 / fanIn) : Math.sqrt(1 / fanIn)
    weights.push(Array.from({ length: sizes[layer] }, () =>
      Array.from({ length: fanIn }, () => normal(rng) * scale),
    ))
    biases.push(Array.from({ length: sizes[layer] }, () => 0))
  }
  return { sizes: [...sizes], hiddenActivation, task, weights, biases }
}

export function cloneNetwork(network) {
  return {
    ...network,
    sizes: [...network.sizes],
    weights: network.weights.map((layer) => layer.map((row) => [...row])),
    biases: network.biases.map((layer) => [...layer]),
  }
}

export function forward(network, input) {
  const activations = [[...input]]
  const zs = []
  let current = input
  network.weights.forEach((weights, layerIndex) => {
    const z = weights.map((row, unit) =>
      row.reduce((sum, weight, source) => sum + weight * current[source], network.biases[layerIndex][unit]),
    )
    zs.push(z)
    const isOutput = layerIndex === network.weights.length - 1
    current = isOutput
      ? (network.task === 'classification' ? softmax(z) : z)
      : z.map((value) => activate(value, network.hiddenActivation))
    activations.push(current)
  })
  return { activations, zs, output: current }
}

function zeroGradients(network) {
  return {
    weights: network.weights.map((layer) => layer.map((row) => row.map(() => 0))),
    biases: network.biases.map((layer) => layer.map(() => 0)),
  }
}

function accumulateGradient(network, x, y, gradients) {
  const pass = forward(network, x)
  const layerCount = network.weights.length
  const deltas = Array(layerCount)
  const prediction = pass.output

  if (network.task === 'classification') {
    deltas[layerCount - 1] = prediction.map((value, index) => value - y[index])
  } else {
    deltas[layerCount - 1] = prediction.map((value, index) => value - y[index])
  }

  for (let layer = layerCount - 1; layer >= 0; layer -= 1) {
    const previousActivation = pass.activations[layer]
    deltas[layer].forEach((delta, unit) => {
      gradients.biases[layer][unit] += delta
      previousActivation.forEach((value, source) => {
        gradients.weights[layer][unit][source] += delta * value
      })
    })

    if (layer > 0) {
      const previousZ = pass.zs[layer - 1]
      const previousA = pass.activations[layer]
      deltas[layer - 1] = previousA.map((activation, source) => {
        const downstream = network.weights[layer].reduce(
          (sum, row, unit) => sum + row[source] * deltas[layer][unit],
          0,
        )
        return downstream * activationDerivative(
          previousZ[source],
          activation,
          network.hiddenActivation,
        )
      })
    }
  }
}

export function trainEpoch(network, rows, learningRate, batchSize, rng) {
  const order = Array.from({ length: rows.length }, (_, index) => index)
  for (let index = order.length - 1; index > 0; index -= 1) {
    const swap = Math.floor(rng() * (index + 1))
    ;[order[index], order[swap]] = [order[swap], order[index]]
  }

  for (let start = 0; start < order.length; start += batchSize) {
    const batch = order.slice(start, start + batchSize)
    const gradients = zeroGradients(network)
    batch.forEach((rowIndex) => {
      const row = rows[rowIndex]
      accumulateGradient(network, row.x, row.y, gradients)
    })
    const scale = learningRate / batch.length
    network.weights.forEach((layer, layerIndex) => {
      layer.forEach((row, unit) => {
        row.forEach((_, source) => {
          network.weights[layerIndex][unit][source] -= gradients.weights[layerIndex][unit][source] * scale
        })
        network.biases[layerIndex][unit] -= gradients.biases[layerIndex][unit] * scale
      })
    })
  }
}

export function evaluate(network, rows) {
  let loss = 0
  let correct = 0
  let absoluteError = 0
  rows.forEach((row) => {
    const prediction = forward(network, row.x).output
    if (network.task === 'classification') {
      loss -= row.y.reduce((sum, target, index) =>
        sum + target * Math.log(Math.max(prediction[index], 1e-9)), 0)
      const predicted = prediction.indexOf(Math.max(...prediction))
      const actual = row.y.indexOf(Math.max(...row.y))
      if (predicted === actual) correct += 1
    } else {
      const error = prediction[0] - row.y[0]
      loss += 0.5 * error * error
      absoluteError += Math.abs(error)
    }
  })
  return {
    loss: loss / rows.length,
    accuracy: network.task === 'classification' ? correct / rows.length : null,
    mae: network.task === 'regression' ? absoluteError / rows.length : null,
  }
}

function oneHot(index, count) {
  return Array.from({ length: count }, (_, item) => item === index ? 1 : 0)
}

function splitRows(rows, trainingCount) {
  return { train: rows.slice(0, trainingCount), validation: rows.slice(trainingCount) }
}

function marsRows() {
  const rng = mulberry32(1107)
  const rows = Array.from({ length: 640 }, () => {
    const prep = rng()
    const weather = rng()
    const payload = clamp(0.65 * rng() + 0.25 * (1 - prep) + normal(rng) * 0.06)
    const atmosphere = clamp(rng())
    const speed = clamp(0.46 * rng() + 0.25 * payload + 0.22 * (1 - atmosphere) + normal(rng) * 0.08)
    const fuel = clamp(0.48 * rng() + 0.38 * prep - 0.16 * payload + normal(rng) * 0.07)
    const turbulence = clamp(0.62 * weather + 0.38 * rng())
    const terrain = clamp(0.38 * weather + 0.62 * rng())
    const engine = clamp(0.42 * rng() + 0.46 * prep + normal(rng) * 0.07)
    const navigation = clamp(0.4 * rng() + 0.48 * prep - 0.12 * weather + normal(rng) * 0.07)
    const timing = rng()
    const idealTiming = 0.72 - 0.38 * atmosphere
    const score = 1.25
      + 1.7 * fuel + 1.55 * engine + 1.35 * navigation
      - 1.45 * speed - 0.95 * turbulence - 0.75 * terrain
      - 1.15 * payload * (1 - fuel)
      - 1.25 * speed * (1 - atmosphere)
      - 1.0 * turbulence * (1 - navigation)
      - 3.4 * Math.abs(timing - idealTiming)
      + normal(rng) * 0.72
    const safe = rng() < sigmoid(score - 0.55) ? 0 : 1
    return { x: [speed, fuel, payload, atmosphere, turbulence, terrain, engine, navigation, timing], y: oneHot(safe, 2) }
  })
  return splitRows(rows, 480)
}

function housingRows() {
  const rng = mulberry32(2209)
  const rows = Array.from({ length: 640 }, () => {
    const prosperity = rng()
    const access = rng()
    const household = rng()
    const upkeep = rng()
    const area = clamp(0.48 * household + 0.52 * rng())
    const bedrooms = clamp(0.62 * area + 0.26 * household + normal(rng) * 0.09)
    const bathrooms = clamp(0.55 * area + 0.25 * prosperity + normal(rng) * 0.1)
    const location = clamp(0.55 * prosperity + 0.25 * access + 0.2 * rng())
    const age = rng()
    const renovation = clamp(0.48 * upkeep + 0.22 * prosperity + 0.3 * rng() - 0.16 * age)
    const transit = clamp(0.58 * access + 0.24 * prosperity + 0.18 * rng())
    const effectiveAge = age * (1 - 0.68 * renovation)
    const layoutFit = 1 - Math.abs(bedrooms - clamp(area * 0.9 + 0.08))
    let price = 0.05 + 0.31 * Math.sqrt(area) + 0.13 * location + 0.13 * area * location
      + 0.08 * Math.max(0, location - 0.62) + 0.07 * renovation + 0.06 * transit
      + 0.055 * bathrooms + 0.035 * bedrooms * layoutFit - 0.15 * effectiveAge
      + 0.035 * normal(rng) * (1 + Math.abs(area - 0.5))
    if (rng() < 0.035) price += normal(rng) * 0.11
    return { x: [area, bedrooms, bathrooms, location, age, renovation, transit], y: [clamp(price)] }
  })
  return splitRows(rows, 480)
}

function biomeRows() {
  const rng = mulberry32(3301)
  const rows = Array.from({ length: 840 }, () => {
    const gravity = rng()
    const geology = clamp(0.34 * gravity + 0.66 * rng())
    const pressure = clamp(0.42 * gravity + 0.58 * rng())
    const radiation = rng()
    const greenhouse = clamp(0.47 * pressure + 0.53 * rng())
    const water = clamp(0.34 * pressure + 0.66 * rng())
    const tilt = rng()
    const albedo = rng()
    const shielding = clamp(0.35 * gravity + 0.28 * geology + 0.37 * rng())
    const temperature = clamp(0.45 * radiation + 0.34 * greenhouse + 0.18 * geology - 0.25 * albedo + 0.16)
    const seasonal = tilt * (1.15 - pressure)
    const scores = [
      2.2 * water + 0.7 * pressure - 2.1 * Math.abs(temperature - 0.55),
      1.8 * temperature + 1.5 * (1 - water) + 0.45 * radiation,
      2.2 * (1 - temperature) + 0.65 * albedo + 0.4 * seasonal,
      1.55 * water + 0.75 * pressure + 0.65 * shielding - 2.6 * Math.abs(temperature - 0.53),
      2.15 * geology + 0.75 * temperature + 0.4 * radiation,
    ].map((score) => score + normal(rng) * 0.45)
    const probabilities = softmax(scores)
    let draw = rng()
    let label = probabilities.length - 1
    for (let index = 0; index < probabilities.length; index += 1) {
      draw -= probabilities[index]
      if (draw <= 0) { label = index; break }
    }
    return { x: [radiation, pressure, water, greenhouse, geology, gravity, tilt, albedo, shielding], y: oneHot(label, 5) }
  })
  return splitRows(rows, 640)
}

function graspRows() {
  const rng = mulberry32(4409)
  const rows = Array.from({ length: 640 }, () => {
    const density = rng()
    const material = rng()
    const irregularity = rng()
    const size = rng()
    const mass = clamp(0.55 * density + 0.35 * size + normal(rng) * 0.07)
    const slipperiness = clamp(0.42 * material + 0.34 * density + 0.24 * rng())
    const fragility = clamp(0.55 * (1 - density) + 0.3 * material + 0.15 * rng())
    const force = rng()
    const confidence = clamp(0.62 * rng() + 0.26 * (1 - irregularity) + normal(rng) * 0.06)
    const alignment = clamp(0.45 * irregularity + 0.35 * (1 - confidence) + 0.2 * rng())
    const angle = rng()
    const required = clamp(0.12 + 0.38 * mass + 0.25 * slipperiness + 0.14 * irregularity)
    const crush = clamp(0.96 - 0.48 * fragility - 0.12 * (1 - size))
    const idealAngle = 0.5 + 0.28 * (irregularity - 0.5)
    const windowScore = 3.6 * Math.min(force - required, crush - force)
    const score = 0.8 + windowScore - 1.15 * alignment - 1.8 * Math.abs(angle - idealAngle)
      + 0.75 * confidence + normal(rng) * 0.65
    const success = rng() < sigmoid(score) ? 0 : 1
    return { x: [mass, size, slipperiness, fragility, irregularity, force, alignment, angle, confidence], y: oneHot(success, 2) }
  })
  return splitRows(rows, 480)
}

export const EXAMPLES = {
  mars: {
    title: 'Mars lander survival', short: 'Lander', task: 'classification', seed: 101,
    featureNames: ['Descent speed', 'Fuel reserve', 'Payload mass', 'Atmosphere', 'Wind turbulence', 'Terrain slope', 'Engine health', 'Navigation', 'Parachute timing'],
    outputNames: ['Safe landing', 'Mission lost'], hidden: [12], activation: 'relu', epochs: 70, batchSize: 32, learningRate: 0.05,
    dataset: marsRows(),
  },
  housing: {
    title: 'Residential sale-price estimation', short: 'Housing', task: 'regression', seed: 202,
    featureNames: ['Floor area', 'Bedrooms', 'Bathrooms', 'Location score', 'Property age', 'Renovation', 'Transit access'],
    outputNames: ['Sale price'], hidden: [12], activation: 'relu', epochs: 80, batchSize: 32, learningRate: 0.045,
    dataset: housingRows(),
  },
  biome: {
    title: 'Alien-planet biome classification', short: 'Biomes', task: 'classification', seed: 303,
    featureNames: ['Stellar radiation', 'Atmospheric pressure', 'Surface water', 'Greenhouse', 'Geological activity', 'Gravity', 'Axial tilt', 'Mineral albedo', 'Magnetic shielding'],
    outputNames: ['Ocean', 'Desert', 'Ice', 'Forest', 'Volcanic'], hidden: [14], activation: 'relu', epochs: 80, batchSize: 32, learningRate: 0.045,
    dataset: biomeRows(),
  },
  grasp: {
    title: 'Robot grasp success', short: 'Robot grasp', task: 'classification', seed: 404,
    featureNames: ['Object mass', 'Object size', 'Slipperiness', 'Fragility', 'Shape irregularity', 'Gripper force', 'Alignment error', 'Approach angle', 'Sensor confidence'],
    outputNames: ['Secure grasp', 'Drop / damage'], hidden: [12], activation: 'relu', epochs: 70, batchSize: 32, learningRate: 0.055,
    dataset: graspRows(),
  },
}

export function parameterCount(sizes) {
  let total = 0
  for (let index = 1; index < sizes.length; index += 1) {
    total += sizes[index] * (sizes[index - 1] + 1)
  }
  return total
}
