import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from paramiko import SSHClient
from scp import SCPClient



def scptoserver(filename):
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect('orxbit@68.183.162.143', port=22)
    with SCPClient(ssh.get_transport()) as scp:
        scp.put(filename)

class Watcher:
    DIRECTORY_TO_WATCH = "models/in"

    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=False)
        self.observer.start()

        try:
            while True:
                time.sleep(5)

        except:
            self.observer.stop()

        self.observer.join()


class Handler(FileSystemEventHandler):
    @staticmethod
    def on_any_event(event):
        if event.is_directory:
            print "Received created event."

        elif event.event_type == 'created':
            scptoserver('models/epoch-model.hdf5')

        elif event.event_type == 'modified':
            scptoserver('models/epoch-model.hdf5')


if __name__ == '__main__':
    w = Watcher()
    w.run()
