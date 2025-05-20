FROM agnohq/python:3.12

ARG USER=app
ARG APP_DIR=/app
ENV APP_DIR=${APP_DIR}

# Create user and home directory
RUN groupadd -g 61000 ${USER} \
  && useradd -g 61000 -u 61000 -ms /bin/bash -d ${APP_DIR} ${USER}

WORKDIR ${APP_DIR}

# Copy requirements.txt
COPY requirements.txt ./

# Install requirements
RUN uv pip sync requirements.txt --system

# Copy project files
COPY . .

# Set permissions for the /app directory
RUN chown -R ${USER}:${USER} ${APP_DIR}

# Make heroku-setup.sh executable
RUN chmod +x scripts/heroku-setup.sh

# Switch to non-root user
USER ${USER}

# Heroku specific: use PORT environment variable
ENV PORT=8000

# Use our setup script to parse DATABASE_URL
CMD ["scripts/heroku-setup.sh"] 