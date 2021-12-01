from confluent_kafka import Producer
import socket

conf = {'bootstrap.servers': "host1:9092, host2:9092", 'client.id': socket.gethostname()}
p = Producer(config=conf)