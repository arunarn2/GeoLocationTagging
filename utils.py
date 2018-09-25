import matplotlib as mpl
import logging
mpl.use('Agg')

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

short_state_names = {
        # 'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        # 'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        # 'GU': 'Guam',
        # 'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        # 'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        # 'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

stop_words = ['the', 'of', 'and', 'to', 'a', 'in', 'for', 'is', 'on', 'that', 'by', 'this', 'with', 'i', 'you',
              'it', 'not', 'or', 'be', 'are', 'from', 'at', 'as', 'your', 'all', 'have', 'new', 'more', 'an', 'was',
              'we', 'will', 'home', 'can', 'us', 'about', 'if', 'page', 'my', 'has', 'search', 'free', 'but', 'our',
              'one', 'other', 'do', 'no', 'information', 'time', 'they', 'site', 'he', 'up', 'may', 'what', 'which',
              'their', 'news', 'out', 'use', 'any', 'there', 'see', 'only', 'so', 'his', 'when', 'contact', 'here',
              'business', 'who', 'web', 'also', 'now', 'help', 'get', 'pm', 'view', 'online', 'c', 'e', 'first',
              'am', 'been', 'would', 'how', 'were', 'me', 's', 'services', 'some', 'these', 'click', 'its', 'like',
              'service', 'x', 'than', 'find', 'price', 'date', 'back', 'top', 'people', 'had', 'list', 'name',
              'just', 'over', 'state', 'year', 'day', 'into', 'email', 'two', 'health', 'n', 'world', 're', 'next',
              'used', 'go', 'b', 'work', 'last', 'most', 'products', 'music', 'buy', 'data', 'make', 'them',
              'should', 'product', 'system', 'post', 'her', 'city', 't', 'add', 'policy', 'number', 'such', 'please',
              'available', 'copyright', 'support', 'message', 'after', 'best', 'software', 'then', 'jan', 'good',
              'video', 'well', 'd', 'where', 'info', 'rights', 'public', 'books', 'high', 'school', 'through', 'm',
              'each', 'links', 'she', 'review', 'years', 'order', 'very', 'privacy', 'book', 'items', 'company', 'r',
              'read', 'group', 'sex', 'need', 'many', 'user', 'said', 'de', 'does', 'set', 'under', 'general',
              'research', 'university', 'january', 'mail', 'full', 'map', 'reviews', 'program', 'life']


dialect_state = {
        'atlantic': ["Connecticut", "Delaware", "Florida", "Georgia", "Maine", "Maryland", "Massachusetts",
                     "New Hampshire", "New Jersey", "New York", "North Carolina", "Pennsylvania", "Rhode Island",
                     "South Carolina", "Vermont", "Virginia", "Washington, DC"],
        'central': ["Arkansas", "Kansas", "Missouri", "Nebraska", "Oklahoma"],
        'central atlantic': ["Delaware", "Washington, DC"],
        'delmarva': ["Delaware"],
        'desert southwest': ["Arizona", "New Mexico"],
        'great lakes': ['michigan', 'minnesota', 'wisconsin'],
        'golf states': ['alabama', 'florida', 'louisiana', 'mississippi'],
        'inland north': ['michigan', 'montana', 'new york', 'washington', 'minnesota', 'north dakota',
                         'oregon', 'wisconsin'],
        'inland south': ["Alabama", "Kentucky", "Mississippi", "Tennessee"],
        'lower mississippi valley': ['arkansas', 'mississippi', 'louisiana'],
        'middle atlantic': ['maryland', 'south carolina', 'washington, dc', 'north carolina', 'virginia'],
        'midland': ['kentucky', 'nebraska', 'tennessee'],
        'mississippi valley': ["Arkansas", "Illinois", "Iowa", "Louisiana", "Minnesota", "Mississippi", "Missouri",
                               "Wisconsin"],
        'mississippi-ohio valley': ["Illinois", "Indiana", "Iowa", "Kentucky", "Minnesota", "Missouri", "Ohio",
                                    "Wisconsin"],
        'new england': ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont"],
        'north': ["Connecticut", "Maine", "Massachusetts", "Michigan", "Minnesota", "Montana", "New Hampshire",
                  "New York", "North Dakota", "Oregon", "Rhode Island", "Vermont", "Washington", "Wisconsin"],
        'north atlantic': ["Connecticut", "Maine", "Massachusetts", "New Hampshir", "Rhode Island", "Vermont"],
        'north central': ["Illinois", "Indiana", "Kentucky", "Michigan", "Ohio", "Wisconsin"],
        'north midland': ['nebraska'],
        'northeast': ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "New Jersey", "New York",
                      "Rhode Island", "Vermont"],
        'northwest': ["Idaho", "Oregon", "Washington", "Montana", "Wyoming"],
        'ohio valley': ['kentucky'],
        'pacific': ['california', 'washington', 'oregon'],
        'pacific northwest': ['washington', 'oregon'],
        'plains states': ['nebraska', 'kansas'],
        'rocky mountains': ['montana', 'utah', 'idaho', 'nevada', 'wyoming'],
        'south': ['florida', 'washington, dc', 'alabama', 'georgia', 'louisiana', 'mississippi', 'north carolina',
                  'south carolina'],
        'south atlantic': ['florida', 'georgia', 'north carolina', 'south carolina'],
        'south midland': ['kentucky', 'arkansas', 'tennessee', 'washington, dc', 'west virginia'],
        'southeast': ['alabama', 'georgia', 'north carolina', 'tennessee', 'north carolina', 'mississippi', 'florida'],
        'southwest': ['arizona', 'new mexico', 'texas', 'oklohama'],
        'upper midwest': ['iowa', 'nebraska', 'south dakota', 'north dakota', 'minnesota'],
        'upper mississippi valley': ['iowa', 'minnesota', 'wisconsin', 'illinois', 'missouri'],
        'west': ["Arizona", "California", "Colorado", "Idaho", "Montana", "Nevada", "New Mexico", "Oregon", "Utah",
                 "Washington", "Wyoming"],
        'west midland': ['iowa', 'ohio', 'arkansas', 'tennessee', 'west virginia', 'illinois', 'indiana', 'kentucky',
                         'nebraska', ]
                 }


def sort_by_value(d):
        items = d.items()
        backitems = [[v[1], v[0]] for v in items]
        backitems.sort(reverse=True)
        return [backitems[i][1] for i in range(0, len(backitems))]


if __name__ == '__main__':
    print("utils")
    # retrieve_location_from_coordinates()
    # get_state_from_coordinates(coordnates=None)
    # contour(coordinates=None, scores=None, world=False, filename='errormap', do_contour=True)
