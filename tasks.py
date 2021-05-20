from celery import Celery
import json

mock_data = [{"car": 4, "person": 2, "bike": 5}, {"car": 9, "person": 1, "bike": 0}]

app = Celery('tasks', broker='amqp://django:djangopass@localhost:5672/celery')

app.send_task('mobility_5g_rest_api.tasks.sensor_fusion', kwargs={'json': mock_data})
