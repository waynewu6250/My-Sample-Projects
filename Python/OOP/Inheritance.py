#Inheritance class


class Employee:

	raise_amt = 1.05
	def __init__(self,first, last, pay):
		self.first = first
		self.last = last
		self.pay = pay
		self.email = first + '.' + last + '@berkeley.edu'
	def raise_amount(self):
		return int(self.pay * self.raise_amt)
	def fullname(self):
		return '{} {}'.format(self.first, self.last)


#Inheritance class:
class Developer(Employee):
	raise_amt = 1.08

	def __init__(self,first, last, pay, prog_lang):

        #Let parent class define first, last, pay
		super().__init__(first, last, pay)
		self.prog_lang = prog_lang

class Manager(Employee):
	def __init__(self,first, last, pay, employees=None):

        #Let parent class define first, last, pay
		super().__init__(first, last, pay)
		if employees is None:
			self.employees = []
		else:
			self.employees = employees
	
	def add_emp(self, emp):
		if emp not in self.employees:
			self.employees.append(emp)

	def remove_emp(self, emp):
		if emp in self.employees:
			self.employess.remove(emp)

	def print_emps(self):
		for emp in self.employees:
			print('-->', emp.fullname())


#
emp1 = Employee('Wayne', 'Wu', 100000)
dev1 = Developer('Wayne', 'Wu', 100000, 'Python')
dev2 = Developer('Sara', 'Wu', 80000, 'Python')
print(emp1.first)
print(emp1.raise_amount())
print(dev1.raise_amount())
print(dev1.prog_lang)

mgr1 = Manager('Sue', 'Smith', 110000, [dev1])
print(mgr1.email)
mgr1.print_emps()
mgr1.add_emp(dev2)
mgr1.print_emps()

print(isinstance(mgr1, Employee))
print(isinstance(mgr1, Developer))
print(issubclass(Manager, Developer))

