import vk_api

from vk_api.longpoll import VkLongPoll, VkEventType, VkMessageFlag

# --
#from commander.commander import Commander
from web.vk_bot import VkBot
# --
import random
import json


def write_msg(user_id, message):
    vk.method('messages.send', {'user_id': user_id, 'message': message, 'random_id': random.randint(0, 100)})


# API-ключ созданный ранее
token = "c6a8bb498207e8649caad369a30e41b5415123d4b3867c78203eb8d1feb60f67835b46f8eddc1dcb1d4e3"

# Авторизуемся как сообщество
vk = vk_api.VkApi(token=token)

# Работа с сообщениями
longpoll = VkLongPoll(vk)

print("Server started")
for event in longpoll.listen():

    if event.type == VkEventType.MESSAGE_NEW:

        if event.to_me:

            print('New message:')
            # vk.method("messages.send", {'peer_id': event.peer_id, 'message': 'Делаю', 
            #                             'random_id': random.randint(0, 100)})

            print(f'For me by: {event.user_id}', end='')

            bot = VkBot(event.user_id)

            
            write_msg(event.user_id, bot.new_message(event.text))

            print('Text: ', event.text)