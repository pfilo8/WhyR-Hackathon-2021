VENUE_DICT = {"sigmod conference": "international conference on management of data",
              "vldb j.": "the vldb journal -- the international journal on very large data bases",
              "vldb": "very large data bases",
              "sigmod record": "acm sigmod record",
              "acm trans . database syst .": "acm transactions on database systems ( tods )"}

VENUE_ENCODE = {"sigmod conference": 10 * 'a',
                "vldb j.": 10 * "b",
                "vldb": 10 * "b",
                "sigmod record": 10 * "c",
                "acm trans . database syst .": 10 * "d"}

OPTIMAL_THRESHOLD = 0.91
