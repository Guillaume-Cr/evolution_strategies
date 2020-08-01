import matplotlib.pyplot as plt

times = [9.382987999999997, 2.549273999999997, 2.5912119999999987, 3.0299949999999995, 3.742715000000004, 2.7572249999999983, 2.0965500000000006, 3.2745340000000027, 2.9413709999999966, 2.622033000000002, 3.305168000000002, 3.270649000000006, 4.52676799999999, 4.304464999999993, 5.078272999999996, 3.5724149999999923, 2.650086999999999, 4.930385999999999, 2.651173, 4.652985000000001, 3.5647170000000017, 4.383748999999995, 3.8510920000000084, 3.7102040000000045, 4.381064999999992, 3.136516999999998, 2.8144790000000057, 3.2423860000000104, 3.1561410000000194, 3.090820000000008, 2.6153440000000217, 2.9306069999999806, 2.7197009999999864, 2.599252000000007, 3.7259369999999876, 2.97186099999999, 2.986532000000011, 3.10503700000001, 3.1296430000000157, 3.650624000000022, 2.8471899999999835, 3.008400999999992, 2.6021639999999877, 3.2200369999999907, 3.7047660000000064, 3.345178000000004, 3.4668550000000096, 3.1191619999999887, 3.817887999999982, 2.9874909999999772, 3.104223999999988, 3.410369000000003, 3.28204199999999, 3.7891189999999995, 3.609841000000017, 3.0837899999999934, 4.048507999999998, 4.25913700000001, 3.6238110000000177, 3.613330999999988, 3.5265559999999994, 3.7565510000000017, 3.3343590000000063, 3.6781380000000183, 3.7088329999999985, 4.057619000000017, 4.309460999999999, 4.343960999999979, 6.13084200000003, 6.896531000000039, 5.809605999999974, 4.864772000000016, 5.014455999999996, 4.677752999999996, 4.9223479999999995, 5.193438000000015, 5.014132000000018, 4.7164960000000065, 4.698232999999959, 3.593150000000037, 3.5945330000000126, 3.300421999999969, 3.701738999999975, 3.8717530000000124, 4.369876999999974, 3.4162830000000213, 4.097765999999979, 4.999937999999986, 4.518803999999989, 3.8650690000000054, 3.9879960000000096, 4.351941000000011, 4.30545699999999, 4.744279000000006, 4.187663999999984, 4.181459000000018, 5.362094000000013, 5.062056000000041, 4.851232999999979, 4.041796999999974, 4.819819999999993, 4.57071000000002, 4.658315000000016, 8.980871000000036, 4.2064209999999775, 4.1167540000000145, 4.576876999999968, 4.724993999999981, 4.29148200000003, 4.076653999999962, 4.585149000000001, 4.500350000000026, 4.013575000000003, 3.8985199999999622, 4.838425000000029, 6.640210000000025, 6.904225999999994, 7.133881999999971, 7.594709000000023, 7.222875000000045, 5.162559999999985, 5.734665999999947, 5.641311999999971, 5.4644289999999955, 4.943548000000078, 5.074618999999984, 5.644443000000024, 6.597460000000069, 5.438265999999999, 5.57696599999997, 6.883327000000008, 6.9774380000000065, 5.776164999999992, 5.349289999999996, 6.267400000000066, 5.73780899999997, 7.2364880000000085, 5.806234000000018, 7.601044999999999, 5.886280000000056, 6.077484000000027, 6.50473199999999, 6.30099800000005, 6.411256999999978, 6.662135000000035, 7.706058000000098, 7.490274999999997, 9.278280999999993, 8.142488000000071, 8.505139999999983, 7.063217000000009, 7.735085000000026, 6.891679000000067, 8.23259900000005, 7.54841799999997, 7.541606000000002, 8.62362399999995, 8.496728000000076, 9.385168000000021, 10.245131000000015, 9.840114000000085, 10.292442000000051, 7.743478999999979, 9.772009000000025, 10.127748999999994, 11.842698000000041, 10.869828999999982, 11.279181999999992, 11.567551999999978, 11.053089, 11.746689999999944, 11.989206000000081, 10.107309999999984, 11.04011700000001, 33.63426300000003, 14.929313000000093, 10.753676000000041, 12.197325999999975, 20.429187999999954, 19.40298699999994, 11.093410999999833, 14.592952000000196, 13.35626000000002, 21.311518000000206, 15.41462300000012, 22.856952000000092, 12.789229999999861, 13.866093000000092, 12.172667000000047, 17.885443000000123, 13.767903999999817, 15.790942000000086, 24.13172200000008, 16.12963200000013, 19.973390999999992, 20.5200319999999, 13.914526000000023, 12.508147000000008, 18.46198500000014, 24.612461000000167, 25.024449000000004, 29.822945000000118, 20.055612999999994, 18.65352999999982, 28.505353000000014, 17.875713000000133, 17.756327000000056, 27.603198999999904, 19.31025999999997, 21.18378000000007, 19.369094000000132, 24.27170500000011, 32.012837999999874, 24.890080000000125, 26.138945999999805, 23.157293999999865, 14.014151000000084, 24.048659000000043, 26.047910999999885, 29.57393400000001, 23.485836999999947, 22.80272500000001, 27.350543000000016, 13.66059099999984, 15.926255000000083, 19.06458900000007, 26.714195000000018, 22.659943000000112, 11.519767000000002, 13.293241999999736, 21.448145000000295, 11.094907000000148, 12.313776999999845, 14.694713999999749, 13.452463999999964, 15.89982000000009, 16.40168500000027, 15.587637000000086, 12.350199999999859, 15.537060999999994, 18.45671900000025, 29.171763999999712, 17.97986200000014, 14.525693000000047, 11.817629000000125, 12.97246000000041, 11.680926, 11.12977600000022]
rewards = [12.0, 10.0, 12.0, 31.0, 55.0, 53.0, 18.0, 24.0, 10.0, 11.0, 25.0, 19.0, 14.0, 19.0, 24.0, 37.0, 33.0, 41.0, 26.0, 37.0, 15.0, 50.0, 24.0, 24.0, 35.0, 21.0, 73.0, 14.0, 20.0, 28.0, 14.0, 47.0, 81.0, 32.0, 85.0, 36.0, 49.0, 13.0, 41.0, 52.0, 34.0, 36.0, 52.0, 70.0, 35.0, 83.0, 62.0, 62.0, 54.0, 62.0, 32.0, 26.0, 29.0, 63.0, 37.0, 45.0, 92.0, 38.0, 46.0, 33.0, 71.0, 39.0, 38.0, 73.0, 48.0, 52.0, 108.0, 38.0, 53.0, 60.0, 93.0, 41.0, 31.0, 39.0, 39.0, 42.0, 100.0, 35.0, 60.0, 21.0, 81.0, 36.0, 36.0, 59.0, 59.0, 70.0, 43.0, 97.0, 89.0, 114.0, 28.0, 54.0, 42.0, 49.0, 85.0, 30.0, 51.0, 81.0, 51.0, 36.0, 65.0, 63.0, 42.0, 51.0, 44.0, 81.0, 52.0, 73.0, 84.0, 57.0, 71.0, 106.0, 40.0, 40.0, 66.0, 87.0, 128.0, 70.0, 121.0, 57.0, 49.0, 23.0, 88.0, 108.0, 129.0, 35.0, 76.0, 84.0, 200.0, 83.0, 86.0, 68.0, 92.0, 179.0, 173.0, 80.0, 78.0, 143.0, 93.0, 129.0, 63.0, 200.0, 64.0, 158.0, 134.0, 200.0, 179.0, 144.0, 200.0, 200.0, 200.0, 200.0, 200.0, 130.0, 73.0, 176.0, 196.0, 200.0, 182.0, 185.0, 184.0, 142.0, 194.0, 181.0, 200.0, 200.0, 193.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 135.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 145.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 186.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0]

fig2 = plt.figure()
ax = fig2.add_subplot(111)
plt.scatter(rewards, times)
plt.ylabel('Time')
plt.xlabel('Reward')
plt.savefig('time_reward_scatter.png')


Episode 1       Average Score: 12.00
Episode 2       Average Score: 15.50
Episode 3       Average Score: 14.33
Episode 4       Average Score: 24.50
Episode 5       Average Score: 25.00
Episode 6       Average Score: 25.67
Episode 7       Average Score: 24.57
Episode 8       Average Score: 26.25
Episode 9       Average Score: 24.33
Episode 10      Average Score: 23.20
Episode 11      Average Score: 22.55
Episode 12      Average Score: 22.75
Episode 13      Average Score: 22.08
Episode 14      Average Score: 21.64
Episode 15      Average Score: 21.80
Episode 16      Average Score: 22.75
Episode 17      Average Score: 23.24
Episode 18      Average Score: 24.06
Episode 19      Average Score: 24.16
Episode 20      Average Score: 24.40
Episode 21      Average Score: 23.95
Episode 22      Average Score: 23.59
Episode 23      Average Score: 23.30 


current weights [ 0.17640523  0.04001572  0.0978738  ... -0.19758757  0.02240672                                             -0.05090435]
1   {50: 18.0, 51: 12.0, 52: 10.0, 53: 30.0, 54: 13.0, 55: 17.0, 56: 16.0, 57: 30.0, 58: 13.0, 59: 39.0, 60: 24.0, 61: 25.0, 62: 13.0, 63: 16.0, 64: 16.0, 65: 15.0, 66: 39.0, 67: 12.0, 68: 16.0, 69: 29.0, 70: 33.0, 71: 16.0, 72: 16.0, 73: 12.0, 74: 9.0, 75: 18.0, 76: 16.0, 77: 19.0, 78: 13.0, 79: 38.0, 80: 16.0, 81: 12.0, 82: 47.0, 83: 17.0, 84: 42.0, 85: 45.0, 86: 10.0, 87: 31.0, 88: 30.0, 89: 48.0, 90: 40.0, 91: 25.0, 92: 22.0, 93: 11.0, 94: 12.0, 95: 34.0, 96: 13.0, 97: 17.0, 98: 15.0, 99: 9.0} 
current weights [ 0.17640523  0.04001572  0.0978738  ... -0.19758757  0.02240672                                             -0.05090435]
1   {50: 38.0, 51: 12.0, 52: 19.0, 53: 18.0, 54: 20.0, 55: 19.0, 56: 24.0, 57: 28.0, 58: 20.0, 59: 15.0, 60: 18.0, 61: 24.0, 62: 40.0, 63: 33.0, 64: 15.0, 65: 13.0, 66: 24.0, 67: 13.0, 68: 13.0, 69: 44.0, 70: 24.0, 71: 19.0, 72: 27.0, 73: 22.0, 74: 46.0, 75: 20.0, 76: 16.0, 77: 27.0, 78: 23.0, 79: 19.0, 80: 21.0, 81: 13.0, 82: 18.0, 83: 16.0, 84: 11.0, 85: 9.0, 86: 35.0, 87: 19.0, 88: 15.0, 89: 37.0, 90: 15.0, 91: 31.0, 92: 32.0, 93: 16.0, 94: 33.0, 95: 29.0, 96: 32.0, 97: 17.0, 98: 12.0, 99: 12.0}
