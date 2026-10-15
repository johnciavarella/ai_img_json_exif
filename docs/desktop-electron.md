# Running as a Desktop App (Mac & Electron)

The web interface is designed with a responsive layout, resizable panels, and keyboard shortcuts (`Cmd/Ctrl+Enter`, `?`), making it ideal to run as a dedicated desktop application window.

## Option 1: Chrome / Brave / Edge App Mode (Fastest)

You can launch the running service (`http://localhost:8090`) in a dedicated window without browser address bars:

### macOS Command:
```sh
open -a "Google Chrome" --args --app="http://localhost:8090"
# Or with Brave:
# open -a "Brave Browser" --args --app="http://localhost:8090"
```

---

## Option 2: Electron Wrapper via Nativefier

You can generate a standalone `.app` bundle for macOS using `nativefier`:

```sh
# Generate a native Mac application pointing to the local port
npx -y nativefier \
  --name "Photo Classifier" \
  --width 1440 \
  --height 900 \
  --single-instance \
  "http://localhost:8090"
```

This creates a `Photo Classifier-darwin-arm64/` (or `darwin-x64`) directory containing `Photo Classifier.app` which can be moved to `/Applications`.

---

## Option 3: Minimal Electron Runner

If creating a custom Electron wrapper script in Node.js:

1. Create a `package.json`:
   ```json
   {
     "name": "photo-organizer-desktop",
     "version": "1.0.0",
     "main": "main.js"
   }
   ```
2. Create `main.js`:
   ```javascript
   const { app, BrowserWindow } = require('electron');

   function createWindow() {
     const win = new BrowserWindow({
       width: 1440,
       height: 900,
       title: "Photo & Video Organizer",
       webPreferences: { nodeIntegration: false }
     });
     win.loadURL('http://localhost:8090');
   }

   app.whenReady().then(createWindow);
   ```
3. Run with `npx electron .`

