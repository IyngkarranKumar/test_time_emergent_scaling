##GIT SETUP
GITHUB_USERNAME="iyngkarrankumar"
GITHUB_EMAIL="iyngkarrankumar@gmail.com"
GITHUB_TOKEN="INSERT TOKEN HERE"
GIT_REPO="https://github.com/IyngkarranKumar/budget_forcing_emergence.git"

echo "Setting up git on $HOSTNAME"

git config --global user.name $GITHUB_USERNAME
git config --global user.email $GITHUB_EMAIL
git config --global credential.helper store
 git config pull.rebase false

touch ~/.git-credentials
echo "https://$GITHUB_USERNAME:$GITHUB_TOKEN@github.com" >> ~/.git-credentials

echo "git setup complete. Now clone a repository"