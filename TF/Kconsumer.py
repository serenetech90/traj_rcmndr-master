import confluent_kafka as ck
from confluent_kafka import Producer
import socket
from ksql import KSQLAPI as kapi

conf = {'bootstrap.servers': "host1:9092, host2:9092", 'client.id': socket.gethostname()}
p = Producer(config=conf)

kclient = kapi('https://localhost:8088')
kclient.create_table(table_name='coordinates_data', columns_type=['ped_x float32', 'ped_y float32'], value_format='JSON',
                     topic='trajectories',
                     key='ped_id')

kclient.query()