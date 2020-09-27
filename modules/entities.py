from enum import Enum
import numpy as np

#实体类
class EntityTracker():

    def __init__(self):
        self.entities = {
                '<hotel_parking>' : None,
                '<hotel_book_people>' : None,
                '<hotel_area>' : None,
                '<hotel_type>' : None,
                '<hotel_price_range>': None,
                '<hotel_book_day>': None,
                '<hotel_stars>': None,
                '<hotel_book_stay>': None,
                '<train_day>': None,
                '<train_departure>': None,
                '<train_book_people>': None,
                '<restaurant_area>': None,
                '<restaurant_price_range>': None,
                '<restaurant_book_time>': None,
                '<restaurant_book_people>': None,
                '<restaurant_food>': None,
                '<restaurant_name>': None,
                '<restaurant_book_day>': None,
                }
        self.num_features = 18 # tracking 4 entities
        self.rating = None

        # constants
        self.hotel_parking = ['yes', "do n't care", 'no', 'free']
        self.hotel_book_people = ['6', '1', '3', '5', '4', '2', '8', '7', 'six', '3.']
        self.hotel_area = ['east', 'north', 'centre', 'south', 'west', "do n't care", 'southern aylesbray', 'cambridge', 'el shaddai', 'stevenage', 'place to be a guesthouse', 'peterborough', 'bishops stortford', 'cheap']
        self.hotel_type = ['hotel', 'guest house']
        self.hotel_price_range = ['cheap', "do n't care", 'moderate', 'expensive']
        self.hotel_book_day = ['tuesday', 'friday', 'monday', 'wednesday', 'saturday', 'thursday', 'sunday']
        self.hotel_stars = ['4', '2', '0', '3', "do n't care", '1', '5']
        self.hotel_book_stay = ['cheap', 'expensive', 'moderate']
        self.train_day = ['friday', 'wednesday', 'monday', 'saturday', 'thursday', 'tuesday', 'sunday', "do n't care", 'fr', 'n', 'we', 'train']
        self.train_departure = ['cambridge', 'birmingham new street', 'ely', 'norwich', 'bishops stortford', 'peterborough', 'stevenage', 'broxbourne', 'london liverpool street', 'leicester', 'stansted airport', 'kings lynn', 'london kings cross', 'birmingham', "do n't care", 'saint johns', 'wandlebury country park', 'liecester', 'panahar', 'cineworld', 'stansted', 'el shaddai', 'lon', 'cafe uno', 'leicaster', 'city hall', "rosa's bed and breakfast", 'norwhich', 'cam', 'brookshite', 'bro', 'cambrid', 'arbu', 'aylesbray lodge guest', 'alpha-milton', 'a', 'london', 'hamilton lodge', 'duxford', 'camboats']
        self.train_book_people = ['1', '5', '8', '6', '2', '7', '4', '3', '0', '9', '15', '`1', '10']
        self.restaurant_area = ['cheap', 'expensive', 'moderate']
        self.restaurant_price_range = ['centre', 'south', 'north', 'east', 'west']
        self.restaurant_book_time = ['19:45', '13:45', '10:15', '19:15', '11:30', '10:30', '18:45', '13:30', '15:00', '11:45', '12:00', '15:15', '16:45', '15:45', '17:15', '19:30', '14:00', '10:45', '17:30', '16:30', '17:00', '12:30', '18:15', '18:00', '20:15', '12:45', '14:15', '13:15', '10:00', '16:00', '19:00', '12:15', '11:00', '11:15', '15:30', '14:30', '18:30', '14:45', '17:45', '09:15', '09:45', '16:15', '13:00', '20:00', '21:00', '20:30', '20:45', '1545', '1745', '09:00', 'not given', "do n't care", '13:10', '21:45', '08:45', '09:30']
        self.restaurant_book_people = ['2', '3', '1', '5', '6', '4', '7', '8']
        self.restaurant_food = ['turkish', 'indian', 'chinese', 'seafood', 'italian', 'british', 'australasian', 'australian', 'asian oriental', 'thai', 'vegetarian', 'modern european', 'gastropub', 'south indian', 'european', 'portuguese', 'swiss', 'crossover', 'catalan', 'french', "do n't care", 'mexican', 'welsh', 'korean', 'tuscan', 'new zealand', 'molecular gastronomy', 'eritrean', 'british food', 'the americas', 'north american', 'spanish', 'barbeque', 'persian', 'greek', 'lebanese', 'vietnamese', 'belgian', 'creative', 'jamaican', 'scottish', 'cuban', 'japanese', 'sri lankan', 'light bites', 'moroccan', 'latin american', 'african', 'basque', 'modern global', 'halal', 'mediterranean', 'bistro', 'international', 'unusual', 'north indian', 'modern eclectic', 'danish', 'afghan', 'world', 'northern european', 'german', 'cantonese', 'irish', 'romanian', 'russian', 'english', 'corsica', 'steakhouse', 'hungarian', 'singaporean', 'austrian', 'venetian', 'ital', 'polynesian', 'kosher', 'swedish', 'scandinavian', 'modern american', 'christmas', 'malaysian', 'north african', 'brazilian', 'canapes', 'caribbean', 'south african', 'traditional', 'indonesian', 'middle eastern', 'fusion', 'polish', 'asian', 'not mentionedc', 'afternoon tea', 'eastern european', 'panasian', 'kor', 'gastro pub', 'american', 'pizza', 'modern  european', 'modern english']
        self.restaurant_name = ['meze bar restaurant', 'indian', 'pizza hut city centre', 'the good luck chinese food takeaway', 'caffe uno', 'the gardenia', 'the oak bistro', "do n't care", 'sala thong', 'thanh binh', 'riverside brasserie', 'cambri', 'pizza express', 'yippee noodle bar', 'curry prince', 'midsummer house restaurant', 'cote', 'restaurant alimentum', 'nandos city centre', 'chiquito restaurant bar', 'maharajah tandoori restaurant', 'yu garden', 'bangkok city', 'copper kettle', 'backstreet bistro', 'the golden curry', 'don pasquale pizzeria', 'sesame restaurant and bar', 'charlie', 'the cow pizza kitchen and bar', 'india house', 'loch fyne', 'eraina', 'royal spice', 'prezzo', 'curry king', 'the nirala', 'curry garden', 'zizzi cambridge', 'da vinci pizzeria', 'jinling noodle bar', 'la raza', 'cotto', 'efes restaurant', 'taj tandoori', 'golden wok', 'charlie chan', 'kohinoor', 'bedouin', 'the cambridge chop house', 'stazione restaurant and coffee bar', 'graffiti', 'pizza hut', 'la mimosa', 'city stop', 'grafton hotel restaurant', 'pizza hut fen ditton', 'ba', 'frankie and bennys', 'rajmahal', 'rice boat', 'the missing sock', 'the varsity restaurant', 'panahar', 'nandos', 'sitar tandoori', 'oak bistro', 'scudamores punt', 'lovel', 'anatolia', 'clowns cafe', 'gourmet burger kitchen', 'tandoori palace', 'ali baba', 'darrys cookhouse and wine shop', 'hakka', 'peking restaurant', 'de luca cucina and bar', 'the slug and lettuce', 'city stop restaurant', 'kymmoy', 'cambridge lodge restaurant', 'tandoori', 'bloomsbury restaurant', 'ugly duckling', 'hk fusion', 'pizza hut cherry hinton', 'fitzbillies restaurant', 'hotel du vin and bistro', 'no', 'restaurant two two', 'dojo noodle bar', 'fi', 'the copper kettle', 'michaelhouse cafe', 'restaurant one seven', 'the hotpot', 'royal standard', 'lev', 'the river bar steakhouse and grill', 'pipasha restaurant', 'golden curry', 'saigon city', 'pizza express fen ditton', 'little seoul', 'meghna', 'saffron brasserie', 'j restaurant', 'la margherita', 'the lucky star', 'lan hong house', 'hotpot', 'ask', 'the gandhi', 'cocum', 'golden house', 'la tasca', 'shanghai family restaurant', 'worth house', 'wagamama', 'galleria', 'lo', 'travellers rest', 'mahal of cambridge', 'archway', 'molecular gastonomy', 'european', 'saint johns chop house', 'anatolia and efes restaurant', 'shiraz restaurant', 'nirala', 'not metioned', 'cott', 'cambridge chop house', 'bridge', 'lucky star', 'clu', 'tang chinese', 'the', 'golden house                            golden house', 'rice house', 'limehouse', 'clowns', 'restaurant', 'parkside pools', 'the dojo noodle bar', 'nusha', 'hobson house', 'au', 'curry queen', 'el shaddai', 'old school', 'el', 'cam', 'yes', 'gardenia', 'fin', 'efes', 'slug and lettuce', 'camboats', 'missing sock', 'grafton', 'nus', 'cambridge lodge', 'fitzbillies', 'hamilton lodge', 'gastropub', 'funky', 'cow pizza', 'ashley', 'ros', 'hobso', 'kitchen and bar', 'd', 'cityr', 'pipasha', 'seasame restaurant and bar', 'the alex', 'hu', 'one seven', 'shanghi family restaurant', 'cambridge be', 'dif', 'margherita', 'ac', 'bri', 'india', 'adden', 'ian hong house']
        self.restaurant_book_day = ['thursday', 'wednesday', 'friday', 'monday', 'sunday', 'saturday', 'tuesday', 'thur', 'not given', 'w']


        self.EntType = Enum('Entity Type', '<hotel_parking> <hotel_book_people> <hotel_area> <hotel_type> <hotel_price_range> <hotel_book_day> <hotel_stars> <hotel_book_stay> <train_day> <train_departure> <train_book_people> <restaurant_area> <restaurant_price_range> <restaurant_book_time> <restaurant_book_people> <restaurant_food> <restaurant_name> <restaurant_book_day>')

#返回word的类型，由5种可能：party_size，location，cuisine，rest_type，传入的word
    def ent_type(self, ent):
        if ent in self.hotel_parking:
            return self.EntType['<hotel_parking>'].name
        elif ent in self.hotel_book_people:
            return self.EntType['<hotel_book_people>'].name
        elif ent in self.hotel_area:
            return self.EntType['<hotel_area>'].name
        elif ent in self.hotel_type:
            return self.EntType['<hotel_type>'].name
        elif ent in self.hotel_price_range:
            return self.EntType['<hotel_price_range>'].name
        elif ent in self.hotel_book_day:
            return self.EntType['<hotel_book_day>'].name
        elif ent in self.hotel_stars:
            return self.EntType['<hotel_stars>'].name
        elif ent in self.hotel_book_stay:
            return self.EntType['<hotel_book_stay>'].name
        elif ent in self.train_day:
            return self.EntType['<train_day>'].name
        elif ent in self.train_departure:
            return self.EntType['<train_departure>'].name
        elif ent in self.train_book_people:
            return self.EntType['<train_book_people>'].name
        elif ent in self.restaurant_area:
            return self.EntType['<restaurant_area>'].name
        elif ent in self.restaurant_price_range:
            return self.EntType['<restaurant_price_range>'].name
        elif ent in self.restaurant_book_time:
            return self.EntType['<restaurant_book_time>'].name
        elif ent in self.restaurant_book_people:
            return self.EntType['<restaurant_book_people>'].name
        elif ent in self.restaurant_food:
            return self.EntType['<restaurant_food>'].name
        elif ent in self.restaurant_name:
            return self.EntType['<restaurant_name>'].name
        elif ent in self.restaurant_book_day:
            return self.EntType['<restaurant_book_day>'].name
        else:
            return ent

#提取话中的单词，并转化为槽位，没有对应的则原话返回：ok let me look into some options for you；api_call <cuisine> <location> <party_size> <rest_type>
    def extract_entities(self, utterance, update=True):
        tokenized = []
        for word in utterance.split(' '):
            entity = self.ent_type(word)
            if word != entity and update:
                self.entities[entity] = word
            tokenized.append(entity)
        return ' '.join(tokenized)


    def context_features(self):
        #keys为键值列表：['<location>', '<rest_type>', '<cuisine>', '<party_size>']
       keys = list(set(self.entities.keys()))
       self.ctxt_features = np.array( [bool(self.entities[key]) for key in keys], 
                                   dtype=np.float32 )
       # print('lalla')
       #
       # print([self.entities[key] for key in keys])
       # print([bool(self.entities[key]) for key in keys])
       # print(self.ctxt_features)
        #输出结果如下：
       #  lalla
       #  [None, None, 'paris', 'italian']
       #  [False, False, True, True]
       #  [0. 0. 1. 1.]

       return self.ctxt_features


    def action_mask(self):
        print('Not yet implemented. Need a list of action templates!')
