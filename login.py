from app import verify_user

def login_user():
    username = input("Enter username: ")
    if verify_user(username):
        print("Login successful!")
    else:
        print("Login failed!")

if __name__ == '__main__':
    login_user()
