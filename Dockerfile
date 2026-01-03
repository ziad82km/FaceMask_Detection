# 1. Start with the base
FROM python:3.10-slim

# 2. Install system libraries (as root)
RUN apt-get update && apt-get install -y libgl1 libglx-mesa0 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# 3. Create the user (Crucial: User must exist BEFORE copying files)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# 4. Copy ONLY requirements first (Optimization: Better caching)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. NOW COPY THE REST (This is where your images and app.py are moved)
# This line MUST be here so the 'user' owns your images (bg.png, etc.)
COPY --chown=user . .

# 6. Finally, start the app
CMD ["streamlit", "run", "app.py", "--server.port=7860"]
