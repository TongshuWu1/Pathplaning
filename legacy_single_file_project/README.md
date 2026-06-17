# Legacy Search-CAGE Single-File Project

This is the old simulator collapsed into one runnable Python file. It is meant to be opened as a separate PyCharm project without depending on the current root `src/` package.

## Run

```bash
python main.py
```

Headless check:

```bash
python main.py --headless --steps 300
```

Useful interactive options:

```bash
python main.py --renderer pygame --fps 60 --width 1920 --height 1080
python main.py --renderer matplotlib --ui-profile fast
```

Install dependencies if needed:

```bash
pip install -r requirements.txt
```
