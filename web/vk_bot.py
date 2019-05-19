import bs4 as bs4
import requests
from ml.model import Model


class VkBot:

    def __init__(self, user_id):
        print(f"Создан объект бота для пользователя {user_id}!")

        self._USER_ID = user_id
        self._USERNAME = self._get_user_name_from_vk_id(user_id)

        self._COMMANDS = ["ПРИВЕТ", "ПОГОДА", "ВРЕМЯ", "ПОКА", "КОНСУЛЬТАЦИЯ", "НАЧАТЬ"]
        self._REGISTED_ID = []
        #self.model = Model()

    def _get_user_name_from_vk_id(self, user_id):
        request = requests.get("https://vk.com/id" + str(user_id))
        bs = bs4.BeautifulSoup(request.text, "html.parser")

        user_name = self._clean_all_tag_from_str(bs.findAll("title")[0])

        return user_name.split()[0]

    def new_message(self, message):
        model = Model()

        return model.get_answer(message)
        # # Привет
        # if message.upper() == self._COMMANDS[0] or message.upper() == self._COMMANDS[5]:
        #     if self._USER_ID not in self._REGISTED_ID:
        #         self._REGISTED_ID.append(self._USER_ID)
        #     return f"Привет-привет, {self._USERNAME}!"

        # # Погода
        # elif message.upper() == self._COMMANDS[1]:
        #     return self._get_weather()

        # # Время
        # elif message.upper() == self._COMMANDS[2]:
        #     return self._get_time()

        # # Пока
        # elif message.upper() == self._COMMANDS[3]:
        #     return f"Пока-пока, {self._USERNAME}!"

        # # Консультация
        # elif message.upper() == self._COMMANDS[4]:
        #     return f"Готов к консультации, {self._USERNAME}!"

        # else:
        #     try:
        #         res1 = "Похожий товар: " + \
        #                str(self.model.most_similar_id(int(message))) + "\n"
        #     except Exception as e:
        #         res1 = "Похожий товар не найден " + str(e) + "\n"
        #     try:
        #         res2 = "Дополнительный товар: " + \
        #                str(self.model.predict_output_id(int(message)))
        #     except Exception as e:
        #         res2 = "Дополнительный товар не найден " + str(e) + "\n"
        #     return res1 + res2

    def _get_time(self):
        request = requests.get("https://my-calend.ru/date-and-time-today")
        b = bs4.BeautifulSoup(request.text, "html.parser")
        return self._clean_all_tag_from_str(str(b.select(".page")[0].findAll("h2")[1])).split()[1]

    @staticmethod
    def _clean_all_tag_from_str(string_line):

        """
        Очистка строки stringLine от тэгов и их содержимых
        :param string_line: Очищаемая строка
        :return: очищенная строка
        """

        result = ""
        not_skip = True
        for i in list(string_line):
            if not_skip:
                if i == "<":
                    not_skip = False
                else:
                    result += i
            else:
                if i == ">":
                    not_skip = True

        return result

    @staticmethod
    def _get_weather(city: str = "москва") -> list:

        request = requests.get("https://sinoptik.com.ru/погода-" + city)
        b = bs4.BeautifulSoup(request.text, "html.parser")

        p3 = b.select('.temperature .p3')
        weather1 = p3[0].getText()
        p4 = b.select('.temperature .p4')
        weather2 = p4[0].getText()
        p5 = b.select('.temperature .p5')
        weather3 = p5[0].getText()
        p6 = b.select('.temperature .p6')
        weather4 = p6[0].getText()

        result = ''
        result = result + ('Утром :' + weather1 + ' ' + weather2) + '\n'
        result = result + ('Днём :' + weather3 + ' ' + weather4) + '\n'
        temp = b.select('.rSide .description')
        weather = temp[0].getText()
        result = result + weather.strip()

        return result

    @staticmethod
    def ml_func():
        return