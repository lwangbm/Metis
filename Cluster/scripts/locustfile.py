from locust import HttpLocust, TaskSet, task

# usage:
# > locust -f get-locustfile.py --host=http://localhost:8000

class ClientTaskSet(TaskSet):
    @task(1)
    def task_default(self):
        """ handle default execution"""
        self.client.get("/")

class WebsiteUser(HttpLocust):
    task_set = ClientTaskSet
    min_wait = 0
    max_wait = 1000
