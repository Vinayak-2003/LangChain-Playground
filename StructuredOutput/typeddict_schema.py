from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

new_person: Person = {
    "name": "vinayak",
    "age": 22
}

new_person_2: Person = {
    "name": "vianyak",
    "age": "22"
}

print(new_person)
print(new_person_2)