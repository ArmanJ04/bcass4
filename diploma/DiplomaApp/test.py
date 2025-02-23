from database import users_collection, results_collection

# Добавление тестового пользователя
test_user = {
    "username": "test_user",
    "email": "test@example.com",
    "password": "hashed_password"
}

# Вставляем данные в коллекцию "users"
insert_result = users_collection.insert_one(test_user)
print(f"✅ Пользователь добавлен с ID: {insert_result.inserted_id}")

# Получаем всех пользователей из базы данных
users = users_collection.find()
print("📌 Список пользователей:")
for user in users:
    print(user)
