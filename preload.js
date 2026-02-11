const { contextBridge } = require('electron')

contextBridge.exposeInMainWorld('electronInfo', {
  chrome: process.versions.chrome,
  node: process.versions.node,
  electron: process.versions.electron,
})