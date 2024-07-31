from app import capture_face_images

def register_user():
    username = input("Enter username: ")
    capture_face_images(username)

if __name__ == '__main__':
    register_user()
