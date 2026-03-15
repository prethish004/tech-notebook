# Python Advanced OOP: Multiple Inheritance & Magic Methods

## Table of Contents
1. [Multiple Inheritance](#multiple-inheritance)
2. [Method Resolution Order (MRO)](#mro)
3. [Magic/Dunder Methods](#magic-methods)
4. [Complete Examples](#complete-examples)

***

## 1. Multiple Inheritance

**One child class inherits from multiple parent classes**

```python
class Flyer:
    def fly(self):
        return "Flying high!"

class Swimmer:
    def swim(self):
        return "Swimming fast!"

class Duck(Flyer, Swimmer):  # Multiple inheritance!
    def quack(self):
        return "Quack quack!"

duck = Duck()
print(duck.fly())    # Flying high!
print(duck.swim())   # Swimming fast!
print(duck.quack())  # Quack quack!
```

**Real-world example:**
```python
class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
    
    def work(self):
        return f"{self.name} is working"

class Manager(Employee):
    def __init__(self, name, salary, team_size):
        super().__init__(name, salary)
        self.team_size = team_size
    
    def manage(self):
        return f"Managing {self.team_size} people"

class Developer(Employee):
    def code(self):
        return f"{self.name} is coding"

class TechLead(Manager, Developer):  # Multiple inheritance!
    pass

tl = TechLead("Alice", 120000, 5)
print(tl.work())   # Alice is working
print(tl.manage()) # Managing 5 people
print(tl.code())   # Alice is coding
```

***

## 2. Method Resolution Order (MRO)

**Python's algorithm for which parent's method to call first**

```python
class A:
    def hello(self):
        return "Hello from A"

class B(A):
    def hello(self):
        return "Hello from B"

class C(A):
    def hello(self):
        return "Hello from C"

class D(B, C):  # Order matters!
    pass

d = D()
print(d.hello())  # Hello from B (B before C in MRO)

# Check MRO
print(D.__mro__)
# (<class '__main__.D'>, <class '__main__.B'>, 
#  <class '__main__.C'>, <class '__main__.A'>, <class 'object'>)
```

**MRO Rules:**
1. Child class first
2. **Left-to-right order** of parents
3. Parent's MRO
4. `object` class last

***

## 3. Magic/Dunder Methods

**Special methods that make classes Pythonic** (`__method__`)

### Common Magic Methods

| Method | Purpose | Example |
|--------|---------|---------|
| `__str__` | `print(obj)` | Human-readable string |
| `__repr__` | `repr(obj)` | Developer representation |
| `__len__` | `len(obj)` | Return length |
| `__getitem__` | `obj[key]` | Index access |
| `__eq__` | `obj1 == obj2` | Equality |
| `__add__` | `obj1 + obj2` | Addition |

### `__str__` vs `__repr__`
```python
class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author
    
    def __str__(self):      # For users
        return f"{self.title} by {self.author}"
    
    def __repr__(self):     # For developers
        return f"Book('{self.title}', '{self.author}')"

book = Book("Python Crash Course", "Eric Matthes")
print(book)         # Python Crash Course by Eric Matthes
print(repr(book))   # Book('Python Crash Course', 'Eric Matthes')
```

### Container-like behavior
```python
class ShoppingCart:
    def __init__(self):
        self.items = []
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        return self.items[index]
    
    def __setitem__(self, index, item):
        self.items[index] = item
    
    def add(self, item):
        self.items.append(item)

cart = ShoppingCart()
cart.add("Apple")
cart.add("Banana")
print(len(cart))        # 2
print(cart[0])          # Apple
cart [geeksforgeeks](https://www.geeksforgeeks.org/python/multiple-inheritance-in-python/) = "Orange"      # Works like a list!
print(cart [geeksforgeeks](https://www.geeksforgeeks.org/python/multiple-inheritance-in-python/))          # Orange
```

### Custom operators
```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __str__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(2, 3)
v2 = Vector(1, 4)
v3 = v1 + v2
print(v3)  # Vector(3, 7)
```

***

## 4. Complete Examples

### Employee System (Multiple Inheritance + Magic Methods)
```python
from abc import ABC, abstractmethod

class Person(ABC):
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __str__(self):
        return f"{self.name}, {self.age} years old"
    
    @abstractmethod
    def work(self):
        pass

class Salaried:
    def __init__(self, salary):
        self.salary = salary
    
    def calculate_pay(self):
        return self.salary

class Hourly:
    def __init__(self, hourly_rate, hours):
        self.hourly_rate = hourly_rate
        self.hours = hours
    
    def calculate_pay(self):
        return self.hourly_rate * self.hours

class Developer(Person, Salaried):
    def __init__(self, name, age, salary):
        Person.__init__(self, name, age)
        Salaried.__init__(self, salary)
    
    def work(self):
        return f"{self.name} is coding"
    
    def __repr__(self):
        return f"Developer({self.name}, salary={self.salary})"

class Freelancer(Person, Hourly):
    def __init__(self, name, age, hourly_rate, hours):
        Person.__init__(self, name, age)
        Hourly.__init__(self, hourly_rate, hours)
    
    def work(self):
        return f"{self.name} is freelancing"
    
    def __eq__(self, other):
        return self.calculate_pay() == other.calculate_pay()

# Usage
dev = Developer("Alice", 30, 80000)
free = Freelancer("Bob", 28, 50, 160)

print(dev)              # Alice, 30 years old
print(dev.work())       # Alice is coding
print(dev.calculate_pay())  # 80000
print(repr(dev))        # Developer(Alice, salary=80000)

print(free.calculate_pay())  # 8000 (50*160)
print(dev == free)      # False
```

### Custom Container (Magic Methods)
```python
class PriorityQueue:
    def __init__(self):
        self._items = []
    
    def __len__(self):
        return len(self._items)
    
    def __getitem__(self, index):
        return self._items[index]
    
    def __setitem__(self, index, priority_item):
        self._items[index] = priority_item
    
    def add(self, item, priority=0):
        self._items.append((priority, item))
        self._items.sort()  # Sort by priority
    
    def __str__(self):
        return f"Queue with {len(self)} items"

pq = PriorityQueue()
pq.add("Low priority task", 3)
pq.add("High priority task", 1)
pq.add("Medium task", 2)

print(pq)           # Queue with 3 items
print(pq[0][1])     # High priority task
print(len(pq))      # 3
```

***

## Quick Reference: Magic Methods

```python
class MagicExample:
    def __init__(self, value): self.value = value
    
    def __str__(self): return f"Value: {self.value}"
    def __repr__(self): return f"MagicExample({self.value})"
    def __len__(self): return len(str(self.value))
    def __call__(self): return self.value * 2
    def __eq__(self, other): return self.value == other.value

obj = MagicExample(42)
print(obj)          # Value: 42
print(repr(obj))    # MagicExample(42)
print(len(obj))     # 2
print(obj())        # 84 (makes object callable!)
```

## Pro Tips

1. **Multiple inheritance order matters** → `Class(A, B)` ≠ `Class(B, A)`
2. **`super()` cooperates** across MRO in multiple inheritance
3. **`__repr__` should be unambiguous** (good for debugging)
4. **`__str__` should be readable** (good for users)
5. **Magic methods make classes** feel like built-in types

**Your repo structure:**
```
tech-notebook/oop/
├── basic/
│   └── README.md (previous OOP basics)
├── advanced/
│   ├── README.md (this guide)
│   ├── multiple_inheritance.py
│   └── magic_methods.py
└── exercises/
```
