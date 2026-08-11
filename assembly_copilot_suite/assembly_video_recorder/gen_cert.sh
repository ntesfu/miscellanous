#!/usr/bin/env bash
# Generate a self-signed TLS certificate for the local network.
# iOS requires HTTPS before it will grant camera access, so this is required.
# The Mac's current LAN IP is baked into the cert's SAN list.
set -e

cd "$(dirname "$0")"
mkdir -p certs

# Detect the primary LAN IPv4 address (en0 = Wi-Fi on most Macs, en1 fallback).
IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo 127.0.0.1)"
echo "Baking LAN IP into certificate: $IP"

cat > certs/openssl.cnf <<EOF
[req]
distinguished_name = dn
x509_extensions = v3
prompt = no
[dn]
CN = ego-recorder.local
[v3]
subjectAltName = @alt
basicConstraints = CA:FALSE
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
[alt]
DNS.1 = localhost
DNS.2 = ego-recorder.local
IP.1  = 127.0.0.1
IP.2  = $IP
EOF

openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout certs/key.pem -out certs/cert.pem \
  -days 825 -config certs/openssl.cnf

echo
echo "Created certs/cert.pem and certs/key.pem for IP $IP"
echo "If your Mac's IP changes, re-run this script."
