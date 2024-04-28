from faker import Faker


class RealisticDataGenerator:
    def __init__(self):
        self.fake = Faker()
        self.data = self.generate_data()

    def generate_data(self):
        return {
            "first_name": self.fake.first_name(),
            "last_name": self.fake.last_name(),
            "date_of_birth": self.fake.date_of_birth().strftime("%Y-%m-%d"),
            "social_security_number": self.fake.ssn(),
            "address": self.fake.street_address(),
            "city": self.fake.city(),
            "state": self.fake.state(),
            "zip_code": self.fake.zipcode(),
            "email": self.fake.email(),
            "phone_number": self.fake.phone_number(),
        }
