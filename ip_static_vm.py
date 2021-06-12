from osmclient import client
import os

HOME = os.getenv("HOME", "/home/ubuntu")

hostname = "10.0.13.252"
myclient = client.Client(host=hostname)
resp = myclient.vnf.list()

for vnf in resp:
    if vnf["vnfd-ref"] == "static_5g_mobility-vnf":
        with open(HOME + "/.bashrc", "a") as bash_file:
            bash_file.write("export RABBITMQ_IP='{}'\n".format(
                vnf["ip-address"]
            ))

os.system(". {}/.bashrc".format(HOME))