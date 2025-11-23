# Self-Hosting Guide - Run Without Render/Heroku

## Requirements
- Computer or VPS (Virtual Private Server) running 24/7
- Public IP address or domain name
- Port forwarding configured on your router

## Method 1: Run on Your Own Computer

### Step 1: Install Production Server
```bash
pip install gunicorn
```

### Step 2: Run with Gunicorn
```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### Step 3: Configure Firewall
Windows:
```powershell
New-NetFirewallRule -DisplayName "Flask App" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

### Step 4: Port Forwarding
1. Open your router settings (usually 192.168.1.1)
2. Find "Port Forwarding" section
3. Forward external port 80 → internal IP:8000
4. Your app will be at: http://YOUR_PUBLIC_IP

### Step 5: Get Your Public IP
```powershell
(Invoke-WebRequest -Uri "https://api.ipify.org").Content
```

## Method 2: Use Free VPS Services

### Oracle Cloud (Free Tier - Forever Free)
1. Sign up: https://www.oracle.com/cloud/free/
2. Create VM instance (Ubuntu)
3. Upload your code
4. Install dependencies
5. Run with gunicorn
6. Access via public IP

### AWS Free Tier (12 months free)
1. Sign up: https://aws.amazon.com/free/
2. Launch EC2 instance
3. Upload code
4. Configure security groups
5. Run app

### Google Cloud Platform ($300 credit)
1. Sign up: https://cloud.google.com/free
2. Create Compute Engine instance
3. Deploy Flask app
4. Configure firewall rules

## Method 3: Use Ngrok (Quick Testing)

### Install Ngrok
1. Download: https://ngrok.com/download
2. Extract ngrok.exe

### Run Your App
```powershell
# Terminal 1: Start Flask
python app.py
```

```powershell
# Terminal 2: Expose with ngrok
ngrok http 5000
```

You'll get a public URL like: `https://abc123.ngrok.io`

**Note**: Ngrok free tier resets URL on restart

## Method 4: Use Your Computer + Dynamic DNS

### Step 1: Get Free Domain
- No-IP: https://www.noip.com/free
- DuckDNS: https://www.duckdns.org/
- FreeDNS: https://freedns.afraid.org/

### Step 2: Install Dynamic DNS Client
Update your domain when your IP changes

### Step 3: Run Flask App
```powershell
# Run in production mode
$env:FLASK_ENV="production"
python app.py
```

### Step 4: Keep It Running
Use `nssm` (Non-Sucking Service Manager):
```powershell
# Download nssm
# Install as Windows Service
nssm install MedicalApp "C:\Python\python.exe" "C:\WebPython\app.py"
nssm start MedicalApp
```

## Recommended: Oracle Cloud Free Tier

**Pros:**
✅ Completely free forever
✅ Always-on server
✅ Public IP included
✅ Full control

**Setup Commands:**
```bash
# On Oracle VM
sudo apt update
sudo apt install python3 python3-pip
git clone https://github.com/Suyash-666/medical-recommendation-system.git
cd medical-recommendation-system
pip3 install -r requirements.txt
export FIREBASE_CREDENTIALS='YOUR_JSON_HERE'
gunicorn -w 4 -b 0.0.0.0:80 app:app
```

## Which Option Do You Prefer?

1. **Ngrok** - Quickest (5 minutes)
2. **Your Computer** - Free but must stay on
3. **Oracle Cloud** - Best free option (30 min setup)
4. **Dynamic DNS** - Good for home server

Tell me which you'd like and I'll help you set it up!
