import telnetlib
import time

class Pipette():
  def __init__(self, ip='192.168.178.57', port='7777'):
    self.ip = ip
    self.port = port
    self.tn = telnetlib.Telnet(ip, port)
  

  def set_mode(self, volume=100000, steps=5, speed_up=8, speed_down=8):
    self.tn.write('{"jsonrpc": "2.0","method": "setMode","params": {"mode": "DIS","vol": '+str(volume)+',"stp": '+str(steps)+',"spu": '+str(speed_up)+',"spd": '+str(speed_down)+',"lcid": -1,"lcchk": 2147483633},"id":12}')


  def step(self):
    self.tn.write(b'{"jsonrpc":"2.0","method":"step","id":4}')


if __name__ == '__main__':
  pipette = Pipette()
  pipette.set_mode(volume=100000, steps=3, speed_up=7, speed_down=8)
  for i in range(10):
    pipette.step()
    time.sleep(2)