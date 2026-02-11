const { app, BrowserWindow } = require('electron/main')
const path = require('node:path')

app.commandLine.appendSwitch('enable-unsafe-webgpu')
app.commandLine.appendSwitch('enable-features', 'Vulkan')

function createWindow () {
  const win = new BrowserWindow({
    width: 1100,
    height: 700,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  })

  const distIndex = path.join(__dirname, 'dist', 'index.html')
  const fs = require('node:fs')

  if (fs.existsSync(distIndex)) {
    win.loadFile(distIndex)
  } else {
    win.loadFile(path.join(__dirname, 'index.html'))
  }

  if (process.argv.includes('--devtools')) {
    win.webContents.openDevTools()
  }
}

app.whenReady().then(() => {
  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})