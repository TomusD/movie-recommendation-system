# movie-recommendation-system
MSc/movie recommendation system using different algorithms

This project is a movie recommendation system implemented in Python using the Flask framework. 
The system uses collaborative filtering techniques, more specifically user-user, item-item, and tag-based recommendations algorithms. 
Additionally, I have added a hybrid recommendation that combines these three methods.

Run from the Terminal where you have saved the file.: "python Main.py"

Example :

  F:\py projects\Τριανταφύλλου_Θανάσης_M111_>python Main.py



After 7-10 minutes you will will see this message : 

	* Serving Flask app 'Main'
	* Debug mode: on
	WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
	* Running on http://127.0.0.1:5000
	Press CTRL+C to quit
	* Restarting with stat
	* Debugger is active!
	* Debugger PIN: 338-800-009


Open a web browser and copy/paste the http address (Example from above : http://127.0.0.1:5000)




Use the recommendation system !

Type a user id and from the drop down list choose which algorithm  and which similiarity metric you want.


(* When you use tag algorithm the input is a movie id instead of a user id.*)

(* In hybrid it takes around 30 - 60 seconds for the recommendations to load.*)
 
(* The program needs Flask framework for the web UI to run so you may need to install it with this command: "pip install Flask" . *)

(* I used Python 3.11.3 for the project. *)
