# Python OOP - Basic Concepts with Code Examples

## Table of Contents
1. [Classes & Objects](#classes--objects)
2. [The `self` Parameter](#the-self-parameter)
3. [Constructor (`__init__`)](#constructor-__init__)
4. [Class vs Instance Attributes](#class-vs-instance-attributes)
5. [Methods](#methods)
6. [4 Pillars of OOP](#4-pillars-of-oop)
7. [Complete Example](#complete-example)

***

## 1. Classes & Objects

**Class** = Blueprint/template  
**Object** = Instance of that blueprint

```python
# Class definition
class Dog:
    pass  # Empty class for now

# Creating objects (instances)
dog1 = Dog()
dog2 = Dog()

print(type(dog1))  # <class '__main__.Dog'>
print(dog1)        # <__main__.Dog object at 0x...>
```

***

## 2. The `self` Parameter

**`self`** refers to "the current instance/object"

```python
class Dog:
    def bark(self):  # self = this specific dog
        print("Woof! I'm", self)

dog1 = Dog()
dog2 = Dog()

dog1.bark()  # Woof! I'm <__main__.Dog object...>
dog2.bark()  # Woof! I'm <__main__.Dog object...>
```

***

## 3. Constructor (`__init__`)

**`__init__`** runs automatically when creating objects

```python
class Dog:
    def __init__(self, name, age):
        self.name = name  # Instance attribute
        self.age = age    # Instance attribute
        print(f"New dog {self.name} created!")

    def bark(self):
        print(f"{self.name} says Woof!")

# Creating objects triggers __init__
dog1 = Dog("Buddy", 3)  # New dog Buddy created!
dog2 = Dog("Max", 5)    # New dog Max created!

dog1.bark()  # Buddy says Woof!
print(dog1.age)  # 3
```

***

## 4. Class vs Instance Attributes

| **Class Attribute** | **Instance Attribute** |
|-------------------|----------------------|
| Shared by ALL objects | Unique to each object |
| Defined outside methods | Defined in `__init__` with `self` |

```python
class Dog:
    species = "Canine"  # Class attribute (shared)
    
    def __init__(self, name):
        self.name = name  # Instance attribute (unique)

dog1 = Dog("Buddy")
dog2 = Dog("Max")

print(dog1.species)  # Canine (shared)
print(dog2.species)  # Canine (shared)
print(dog1.name)     # Buddy (unique)
print(dog2.name)     # Max (unique)

# Modify class attribute
Dog.species = "Mammal"
print(dog1.species)  # Mammal
print(dog2.species)  # Mammal
```

***

## 5. Methods

**Instance methods** (use `self`) | **Class methods** | **Static methods**
---|---|---
Operate on object data | Operate on class | Utility functions

```python
class Dog:
    species = "Canine"
    
    def __init__(self, name, breed):
        self.name = name
        self.breed = breed
    
    # Instance method
    def describe(self):
        return f"{self.name} is a {self.breed}"
    
    # Class method (use cls)
    @classmethod
    def get_species(cls):
        return cls.species
    
    # Static method (no self/cls)
    @staticmethod
    def make_sound():
        return "Woof!"

dog = Dog("Buddy", "Golden Retriever")
print(dog.describe())      # Buddy is a Golden Retriever
print(Dog.get_species())   # Canine
print(Dog.make_sound())    # Woof!
```

***

## 6. 4 Pillars of OOP

### **Encapsulation** (Bundle data + methods)

```python
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.__balance = balance  # Private (by convention)
    
    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            print(f"Deposited ${amount}")
    
    def get_balance(self):
        return self.__balance

account = BankAccount("Alice", 1000)
account.deposit(500)
print(account.get_balance())  # 1500
# print(account.__balance)  # Works but don't do this!
```

### **Inheritance** (Child classes inherit from Parent)

```python
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        pass

class Dog(Animal):  # Dog inherits from Animal
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

dog = Dog("Buddy")
cat = Cat("Whiskers")
print(dog.speak())  # Buddy says Woof!
print(cat.speak())  # Whiskers says Meow!
```

### **Polymorphism** (Same method, different behavior)

```python
def make_animal_speak(animal):
    print(animal.speak())

make_animal_speak(dog)  # Buddy says Woof!
make_animal_speak(cat)  # Whiskers says Meow!
```

### **Abstraction** (Hide complexity)

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14 * self.radius ** 2

circle = Circle(5)
print(circle.area())  # 78.5
```

***

## 7. Complete Example: Library System

```python
class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author
        self.is_borrowed = False
    
    def __str__(self):
        status = "Borrowed" if self.is_borrowed else "Available"
        return f"{self.title} by {self.author} [{status}]"

class Library:
    def __init__(self):
        self.books = []
    
    def add_book(self, book):
        self.books.append(book)
        print(f"Added: {book}")
    
    def lend_book(self, title):
        for book in self.books:
            if book.title == title and not book.is_borrowed:
                book.is_borrowed = True
                print(f"Lent: {book}")
                return True
        print(f"Book '{title}' not available")
        return False
    
    def display_books(self):
        print("\n--- Library Books ---")
        for book in self.books:
            print(book)

# Usage
lib = Library()
lib.add_book(Book("Python Crash Course", "Eric Matthes"))
lib.add_book(Book("Clean Code", "Robert Martin"))

lib.display_books()
lib.lend_book("Python Crash Course")
lib.display_books()
```

**Output:**
```
Added: Python Crash Course by Eric Matthes [Available]
Added: Clean Code by Robert Martin [Available]

--- Library Books ---
Python Crash Course by Eric Matthes [Available]
Clean Code by Robert Martin [Available]
Lent: Python Crash Course by Eric Matthes [Borrowed]

--- Library Books ---
Python Crash Course by Eric Matthes [Borrowed]
Clean Code by Robert Martin [Available]
```

***

## Quick Reference Table

| Concept | Keyword/Code | Purpose |
|---------|-------------|---------|
| Class | `class MyClass:` | Blueprint |
| Object | `obj = MyClass()` | Instance |
| Constructor | `def __init__(self):` | Initialize object |
| Instance attr | `self.name = value` | Object-specific data |
| Class attr | `Class.var = value` | Shared by all objects |
| Instance method | `def method(self):` | Works on object |
| Inheritance | `class Child(Parent):` | Reuse code |
| `@classmethod` | `@classmethod` | Works on class |
| `@staticmethod` | `@staticmethod` | Utility function |

## Your Repo Structure
```
tech-notebook/oop/
├── README.md (this guide)
├── examples/
│   ├── 01_classes_objects.py
│   ├── 02_bank_account.py
│   └── 03_library_system.py
└── exercises/
    └── practice.py
```
