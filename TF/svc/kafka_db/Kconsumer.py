import confluent_kafka as ck

from ksql import KSQLAPI as kapi

kclient = kapi('https://localhost:8088')
kclient.create_table(table_name='coordinates_data', columns_type=['ped_x float32', 'ped_y float32'],
                     value_format='JSON',
                     topic='trajectories',
                     key='ped_id')

kclient.query('select * from coordinates_data')
