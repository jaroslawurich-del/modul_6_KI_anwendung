import asyncio

try:
    asyncio.get_running_loop()
    print("Es läuft bereits eine Event-Loop!")
except RuntimeError:
    print("Keine Event-Loop aktiv.")