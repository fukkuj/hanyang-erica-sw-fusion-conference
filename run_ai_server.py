import sys
from ai_server.main import main, handler
from signal import signal, SIGINT


signal(SIGINT, handler)

main(sys.argv)
