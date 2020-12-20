FROM gitpod/workspace-full

USER gitpod
RUN curl https://cli-assets.heroku.com/install.sh 13 | sh
