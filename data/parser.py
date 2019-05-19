import xml.etree.ElementTree as ET
import pandas as pd


class XML2DataFrame:

    def __init__(self, xml_data):
        self.root = ET.XML(xml_data)

    def parse_root(self, root):
        # for child in root.getchildren():
        #     try:
        #         print(child.attrib['Title'])
        #     except KeyError:
        #         continue
        return [self.parse_element(child) for child in root.getchildren()]

    def parse_element(self, element, parsed=None):
        if parsed is None:
            parsed = dict()

        for key in element.keys():
            if key not in parsed:
                parsed[key] = element.attrib.get(key)


        for child in list(element):
            self.parse_element(child, parsed)
        return parsed

    def process_data(self):
        structure_data = self.parse_root(self.root)
        print(structure_data[0])
        return pd.DataFrame(structure_data)

with open('/home/igor/PycharmProjects/TechAtomAnswers/data/Posts.xml', 'rb') as data:
    xml2df = XML2DataFrame(data.read())
xml_dataframe = xml2df.process_data()

print(xml_dataframe.columns)