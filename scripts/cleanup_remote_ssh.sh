#!/bin/bash


#FOR LOCAL
# Close Cursor completely first
# Clear Remote SSH cache
rm -rf ~/.cursor/extensions/ms-vscode-remote.remote-ssh-*/
# Clear connection cache
rm -rf ~/.cursor/User/globalStorage/ms-vscode-remote.remote-ssh/
# Clear workspace cache
rm -rf ~/.cursor/User/workspaceStorage/



#FOR REMOTE
pkill -f "cursor-server\|vscode-server\|code-server"
# Remove server installations
rm -rf ~/.cursor-server*
rm -rf ~/.vscode-server*
# Check for any remaining processes
ps aux | grep -E "(cursor|vscode|code)-server"


###NUCLEAR OPTION###

#Remote 
rm -rf ~/.vscode-server*
rm -rf ~/.cursor-server*
rm -rf ~/.cache/vscode*
rm -rf ~/.cache/cursor*


#local 
# Close Cursor completely first
# Then nuke the entire remote SSH cache
rm -rf ~/.cursor/extensions/ms-vscode-remote.remote-ssh-*
rm -rf ~/.cursor/User/globalStorage/ms-vscode-remote.*
rm -rf ~/.cursor/User/workspaceStorage/
rm -rf ~/.cursor/logs/
rm -rf ~/.cursor/CachedExtensions/

# Also clear VS Code cache if you have it
rm -rf ~/.vscode/extensions/ms-vscode-remote.remote-ssh-*
rm -rf ~/.vscode/User/globalStorage/ms-vscode-remote.*