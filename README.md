# Computer-Vision

Object tracking and detection for cameras in Ria Ativa and Praia da Barra.

This project includes features like:
- Detection of vehicles ( car, truck, motorcycle)
- Detection of person and bicycles in bike lains
- Detection of stopped objects
- Detection of vehicles outside the road
- Detection of animals and strange objects

## How to install

Make sure you are running Python 3.8 or higher

1. Create a virtual environment (venv)
```bash
python3 -m venv venv
```

2. Activate the virtual environment (you need to repeat this step, and this step only, every time you start a new terminal/session):
```bash
source venv/bin/activate
```

3. Install the game requirements:
```bash
pip install -r requirements.txt
```

## How to run the application

Run detection for camera in Ria Ativa 

```bash
python3 -m main.py --cam riaAtiva
```

Run detection for camera Praia da Barra

```bash
python3 -m main.py --cam ponteBarra
```

