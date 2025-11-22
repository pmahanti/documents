# Deployment Guide - Lunar Surface Missions Explorer

## Quick Start

### Option 1: Local Testing (Immediate)
```bash
# Navigate to the project directory
cd /home/user/documents

# Open in browser (choose one):
# Linux:
xdg-open index.html

# macOS:
open index.html

# Windows:
start index.html
```

### Option 2: Simple HTTP Server (Recommended for Testing)

**Python 3:**
```bash
cd /home/user/documents
python3 -m http.server 8000
# Visit: http://localhost:8000
```

**Python 2:**
```bash
cd /home/user/documents
python -m SimpleHTTPServer 8000
# Visit: http://localhost:8000
```

**Node.js (with http-server):**
```bash
cd /home/user/documents
npx http-server -p 8000
# Visit: http://localhost:8000
```

**PHP:**
```bash
cd /home/user/documents
php -S localhost:8000
# Visit: http://localhost:8000
```

## Production Deployment

### GitHub Pages (Free)

1. **Create a new GitHub repository**
   ```bash
   cd /home/user/documents
   git init
   git add index.html styles.css app.js missions_data.json README.md
   git commit -m "Initial commit: Lunar Missions Explorer"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/lunar-missions-explorer.git
   git push -u origin main
   ```

2. **Enable GitHub Pages**
   - Go to repository Settings
   - Navigate to Pages section
   - Select "main" branch as source
   - Click Save
   - Your site will be live at: `https://YOUR_USERNAME.github.io/lunar-missions-explorer/`

### Netlify (Free, Simple)

1. **Drag and Drop Method**
   - Visit https://app.netlify.com/drop
   - Drag the entire project folder
   - Get instant deployment URL

2. **Netlify CLI Method**
   ```bash
   npm install -g netlify-cli
   cd /home/user/documents
   netlify deploy --prod
   ```

### Vercel (Free)

```bash
npm i -g vercel
cd /home/user/documents
vercel --prod
```

### AWS S3 + CloudFront

1. **Create S3 Bucket**
   ```bash
   aws s3 mb s3://lunar-missions-explorer
   aws s3 website s3://lunar-missions-explorer/ --index-document index.html
   ```

2. **Upload Files**
   ```bash
   aws s3 sync /home/user/documents s3://lunar-missions-explorer/ \
       --exclude ".git/*" \
       --exclude "*.md" \
       --acl public-read
   ```

3. **Configure CloudFront** (optional, for HTTPS and CDN)
   - Create CloudFront distribution
   - Point to S3 bucket
   - Configure SSL certificate

### Traditional Web Hosting (cPanel, etc.)

1. **Via FTP/SFTP:**
   - Connect to your web server
   - Upload all files to public_html or web root:
     - index.html
     - styles.css
     - app.js
     - missions_data.json

2. **File Permissions:**
   ```bash
   chmod 644 index.html styles.css app.js missions_data.json
   ```

### Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM nginx:alpine
COPY index.html /usr/share/nginx/html/
COPY styles.css /usr/share/nginx/html/
COPY app.js /usr/share/nginx/html/
COPY missions_data.json /usr/share/nginx/html/
EXPOSE 80
```

Build and run:
```bash
docker build -t lunar-missions-explorer .
docker run -d -p 8080:80 lunar-missions-explorer
# Visit: http://localhost:8080
```

### Nginx Configuration

If deploying to Nginx server:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    root /var/www/lunar-missions-explorer;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    # Cache static assets
    location ~* \.(css|js|json)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Gzip compression
    gzip on;
    gzip_types text/css application/javascript application/json;
}
```

## Offline Package Creation

### Create ZIP Archive for Distribution

```bash
cd /home/user/documents
zip -r lunar-missions-explorer.zip \
    index.html \
    styles.css \
    app.js \
    missions_data.json \
    README.md
```

### Create Standalone Executable (Electron - Optional)

For creating a desktop application:

1. **Install Electron Packager**
   ```bash
   npm install -g electron-packager
   ```

2. **Create package.json**
   ```json
   {
     "name": "lunar-missions-explorer",
     "version": "1.0.0",
     "main": "main.js"
   }
   ```

3. **Create main.js**
   ```javascript
   const { app, BrowserWindow } = require('electron');

   function createWindow() {
       const win = new BrowserWindow({
           width: 1200,
           height: 800,
           webPreferences: {
               nodeIntegration: false,
               contextIsolation: true
           }
       });
       win.loadFile('index.html');
   }

   app.whenReady().then(createWindow);
   ```

4. **Package**
   ```bash
   electron-packager . lunar-missions-explorer --platform=all --arch=x64
   ```

## CDN Optimization (Optional)

If you want to serve from multiple global locations, consider using:

- **Cloudflare**: Free CDN with SSL
- **AWS CloudFront**: Global CDN
- **Fastly**: High-performance CDN

## Performance Optimization Checklist

- ✅ Minify CSS (optional)
- ✅ Minify JavaScript (optional)
- ✅ Compress JSON data
- ✅ Enable gzip/brotli compression on server
- ✅ Set appropriate cache headers
- ✅ Use HTTP/2 if available
- ✅ Enable HTTPS with SSL certificate

## Security Considerations

### Content Security Policy
Add to `index.html` `<head>`:
```html
<meta http-equiv="Content-Security-Policy"
      content="default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'">
```

### HTTPS Enforcement
```nginx
# Nginx redirect
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

## Testing Checklist

Before deployment, verify:
- ✅ All files load correctly
- ✅ Search functionality works
- ✅ All filters work independently and together
- ✅ Mission modals open and close properly
- ✅ Responsive design works on mobile
- ✅ No console errors
- ✅ Works offline (after initial load)
- ✅ Cross-browser compatibility

## Monitoring & Analytics (Optional)

Add Google Analytics or similar:
```html
<!-- Add before </head> in index.html -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
```

## Troubleshooting

### CORS Issues (Local Development)
If JSON won't load locally, use a local server (see Quick Start above).

### Large File Size
- JSON is already optimized (~40KB)
- Consider minifying CSS/JS for production
- Enable server compression (gzip/brotli)

### Browser Compatibility
Test in:
- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers

## Update Procedure

To update mission data:
1. Edit `missions_data.json`
2. Validate JSON syntax
3. Test locally
4. Deploy updated file
5. Clear browser cache if needed

---

**Support**: For deployment issues, check browser console and server logs.
