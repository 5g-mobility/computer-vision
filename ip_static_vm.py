from osmclient import client

hostname = "10.0.13.252"
myclient = client.Client(host=hostname)
resp = myclient.vnf.list()

for vnf in resp:
    if vnf["vnfd-ref"] == "static_5g_mobility-vnf":
        print(vnf["ip-address"])