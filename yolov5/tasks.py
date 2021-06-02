from celery import Celery
import json


class CeleryTasks:

    def __init__(self, rabbit_url='localhost', port=5672, user='django', pw='djangopass', vhost='celery'):
        # 'tasks' is the name of the current module
        self.app = Celery('tasks', broker='amqp://{}:{}@{}:{}/{}'.format(user, pw, rabbit_url, port, vhost))

    def send_data(self, json):
        print("Sending JSON: {}".format(json))
        self.app.send_task('mobility_5g_rest_api.tasks.sensor_fusion', kwargs={'json': json})





