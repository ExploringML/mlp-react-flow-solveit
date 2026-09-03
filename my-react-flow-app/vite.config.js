import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const port = Number(process.env.SOLVEIT_PORT || 8000)
const domains = JSON.parse(process.env.PUBLIC_DOMAINS || '{}')
const publicHost = domains[String(port)]

const tutorialReload = {
  name: 'solveit-tutorial-reload',
  handleHotUpdate({ file, server }) {
    if (!file.includes('/src/')) return
    server.ws.send({ type: 'full-reload' })
    return []
  },
}

export default defineConfig({
  plugins: [react(), tutorialReload],
  server: {
    host: '0.0.0.0',
    port,
    strictPort: true,
    ...(publicHost ? { allowedHosts: [publicHost] } : {}),
    ...(publicHost ? { ws: { protocol: 'wss', host: publicHost, clientPort: 443 } } : {}),
  },
})
