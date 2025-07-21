#!/bin/bash

# HTTPS Setup Script for iSpeak Backend (Fixed Version)
# Run this script on your VPS server

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up HTTPS for iSpeak Backend (Fixed Version)${NC}"

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo -e "${RED}This script should not be run as root${NC}"
   exit 1
fi

# Prompt for domain name
read -p "Enter your domain name (e.g., api.therapy-science.ch): " DOMAIN
read -p "Enter your email for Let's Encrypt notifications: " EMAIL
read -p "Enter the full path to your iSpeak-backend directory: " PROJECT_PATH
read -p "Enter your username: " USERNAME

echo -e "${YELLOW}Domain: $DOMAIN${NC}"
echo -e "${YELLOW}Email: $EMAIL${NC}"
echo -e "${YELLOW}Project Path: $PROJECT_PATH${NC}"
echo -e "${YELLOW}Username: $USERNAME${NC}"

read -p "Is this information correct? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Exiting..."
    exit 1
fi

# Wait for any existing apt processes to finish
echo -e "${GREEN}Waiting for package manager to be available...${NC}"
while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
    echo "Waiting for other package managers to finish..."
    sleep 5
done

# Update system
echo -e "${GREEN}Updating system packages...${NC}"
sudo apt update && sudo apt upgrade -y

# Install required packages
echo -e "${GREEN}Installing Nginx and Certbot...${NC}"
sudo apt install nginx certbot python3-certbot-nginx -y

# Stop nginx if running
sudo systemctl stop nginx || true

# Create temporary HTTP-only Nginx configuration
echo -e "${GREEN}Creating temporary HTTP-only Nginx configuration...${NC}"
sudo tee /etc/nginx/sites-available/ispeak-backend-temp > /dev/null <<EOF
server {
    listen 80;
    server_name $DOMAIN;

    client_max_body_size 100M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    # Let's Encrypt challenge location
    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }
}
EOF

# Enable the temporary site
echo -e "${GREEN}Enabling temporary Nginx site...${NC}"
sudo rm -f /etc/nginx/sites-enabled/default  # Remove default site
sudo rm -f /etc/nginx/sites-enabled/ispeak-backend  # Remove any existing config
sudo ln -sf /etc/nginx/sites-available/ispeak-backend-temp /etc/nginx/sites-enabled/

# Test Nginx configuration
echo -e "${GREEN}Testing temporary Nginx configuration...${NC}"
sudo nginx -t

# Create systemd service
echo -e "${GREEN}Creating systemd service...${NC}"
sudo tee /etc/systemd/system/ispeak-backend.service > /dev/null <<EOF
[Unit]
Description=iSpeak FastAPI Backend
After=network.target

[Service]
Type=exec
User=$USERNAME
Group=$USERNAME
WorkingDirectory=$PROJECT_PATH
ExecStart=/usr/local/bin/poetry run uvicorn iSpeak.api:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=$PROJECT_PATH/data

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable service
echo -e "${GREEN}Setting up systemd service...${NC}"
sudo systemctl daemon-reload
sudo systemctl enable ispeak-backend
sudo systemctl start ispeak-backend

# Check service status
echo -e "${GREEN}Checking service status...${NC}"
sudo systemctl status ispeak-backend --no-pager

# Start Nginx with HTTP-only configuration
echo -e "${GREEN}Starting Nginx with HTTP-only configuration...${NC}"
sudo systemctl start nginx
sudo systemctl enable nginx

# Test if domain resolves and service is accessible
echo -e "${GREEN}Testing domain resolution and HTTP access...${NC}"
if curl -f -s "http://$DOMAIN/" > /dev/null; then
    echo -e "${GREEN}✓ Domain is accessible via HTTP${NC}"
else
    echo -e "${YELLOW}⚠ Warning: Could not access http://$DOMAIN/ - check DNS configuration${NC}"
    echo -e "${YELLOW}Make sure your domain's A record points to this server's IP$(NC)"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Please configure DNS first, then re-run this script"
        exit 1
    fi
fi

# Obtain SSL certificate
echo -e "${GREEN}Obtaining SSL certificate from Let's Encrypt...${NC}"
sudo certbot --nginx -d $DOMAIN --email $EMAIL --agree-tos --non-interactive

# The certbot command above should automatically update the Nginx configuration
# But let's create our own optimized HTTPS configuration

echo -e "${GREEN}Creating optimized HTTPS Nginx configuration...${NC}"
sudo tee /etc/nginx/sites-available/ispeak-backend > /dev/null <<EOF
server {
    listen 80;
    server_name $DOMAIN;
    return 301 https://\$server_name\$request_uri;
}

server {
    listen 443 ssl http2;
    server_name $DOMAIN;

    ssl_certificate /etc/letsencrypt/live/$DOMAIN/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$DOMAIN/privkey.pem;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;

    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;

    client_max_body_size 100M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
EOF

# Replace the temporary configuration with the HTTPS one
echo -e "${GREEN}Enabling HTTPS configuration...${NC}"
sudo rm -f /etc/nginx/sites-enabled/ispeak-backend-temp
sudo ln -sf /etc/nginx/sites-available/ispeak-backend /etc/nginx/sites-enabled/

# Test the new configuration
echo -e "${GREEN}Testing HTTPS Nginx configuration...${NC}"
sudo nginx -t

# Reload Nginx with new configuration
echo -e "${GREEN}Reloading Nginx with HTTPS configuration...${NC}"
sudo systemctl reload nginx

# Test automatic renewal
echo -e "${GREEN}Testing automatic certificate renewal...${NC}"
sudo certbot renew --dry-run

# Configure firewall if ufw is available
if command -v ufw &> /dev/null; then
    echo -e "${GREEN}Configuring firewall...${NC}"
    sudo ufw allow 'Nginx Full'
    sudo ufw allow ssh
    sudo ufw --force enable
fi

# Test HTTPS access
echo -e "${GREEN}Testing HTTPS access...${NC}"
if curl -f -s "https://$DOMAIN/" > /dev/null; then
    echo -e "${GREEN}✓ HTTPS is working correctly!${NC}"
else
    echo -e "${YELLOW}⚠ Warning: HTTPS test failed, but setup may still be correct${NC}"
fi

echo -e "${GREEN}Setup complete!${NC}"
echo -e "${GREEN}Your API should now be available at: https://$DOMAIN${NC}"
echo ""
echo -e "${GREEN}You can check the status of your services with:${NC}"
echo "sudo systemctl status ispeak-backend"
echo "sudo systemctl status nginx"
echo ""
echo -e "${GREEN}View logs with:${NC}"
echo "sudo journalctl -u ispeak-backend -f"
echo "sudo tail -f /var/log/nginx/error.log"
echo ""
echo -e "${GREEN}Test your API:${NC}"
echo "curl https://$DOMAIN/"

# Clean up temporary files
sudo rm -f /etc/nginx/sites-available/ispeak-backend-temp 