#Python Object-Oriented Programming
#Class Vairable, Instance Variable
#Class method, Instance method, Static method

class Employee:

	#class variables
	num_of_emps = 0

	# self represents instane(object) itself, instance variables
	def __init__(self, first, last, pay, raise_amount):
		self.first = first
		self.last = last
		self.pay = pay
		self.raise_amount = raise_amount
		self.email = first + '.' + last + '@company.com'

		Employee.num_of_emps += 1

	#Special Method
	def __repr__(self):
		return "Employee('{}', '{}', '{})".format(self.first, self.last, self.pay)
	def __str__(self):
		return '{} - {}'.format(self.fullname(), self.email)
	def __add__(self, other):
		return self.pay + other.pay
	def __len__(self):
		return len(self.fullname())

	# Define instance methods (function)
	def fullname(self):
		return '{} {}'.format(self.first, self.last)

	def apply_raise(self):
		self.pay = int(self.pay * self.raise_amount)
		return self.pay

	# Define class methods
	@classmethod
	def set_raise_amount(cls, raise_amount):
		cls.raise_amount = raise_amount

	@classmethod
	def from_string(cls, emp):
		first, last, pay, raise_amount = emp.split('-')
		return cls(first, last, pay, raise_amount)

	# Define static methods
	@staticmethod
	def is_workday(day):
		if day.weekday() == 5 or day.weekday() == 6:
			return False
		return True



emp_1 = Employee('Wayne', 'Wu', 100000, 1.05)
emp_info_2 = 'Erica-Chen-100000-1.03'
emp_2 = Employee.from_string(emp_info_2)
emp_3 = Employee('Frank', 'Shyu', 110000, 1.04)


print(emp_1.email)
print(emp_1.raise_amount)
print(emp_1.fullname())
print(emp_1.apply_raise())
print(emp_2.email)

#Print parameters
print(emp_1.__dict__)
print(Employee.num_of_emps)

Employee.set_raise_amount(1.08)
print(Employee.raise_amount)

import datetime
my_date = datetime.date(2018, 7, 10)
print(Employee.is_workday(my_date))

print(emp_1 + emp_3)
print(len(emp_1))

