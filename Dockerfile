# Usa un'immagine Python ufficiale come base
FROM python:3.9-slim

# Imposta la directory di lavoro all'interno del container
WORKDIR /app

# Copia il file delle dipendenze e installale
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il resto dei file dell'app
COPY . .

# Esponi la porta standard di Streamlit
EXPOSE 8501

# Comando per avviare l'app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
