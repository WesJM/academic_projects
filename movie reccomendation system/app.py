from flask import Flask, render_template, request
import recco_sys

app = Flask(__name__)

# Get movie title list
movie_titles_list = recco_sys.movie_list()
movie_images_list = recco_sys.images_list()

# Initialize page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', movies=movie_titles_list, display_error=0)


# Fetch selected movie and similar movies
@app.route('/', methods=['GET', 'POST'])
def get_recommendations():
    selected_movie = request.form.get('movies')
    
    try:
        # Verify input 
        if selected_movie != "":

            # Set movie title to lowercase
            selected_movie = selected_movie.lower()
            # Set all titles in list to lowercase 
            lower_titles = [m.lower() for m in movie_titles_list]

            # Get movie and image from input
            selected_index = lower_titles.index(selected_movie)
            selected_movie_image = movie_images_list[selected_index]
            
            # Function call to get recommendations
            movie_reccos_tuples = recco_sys.movie_recommendations(selected_movie)

            # Get proper movie title for input return
            selected_movie = movie_titles_list[selected_index]

            # Split tuples
            reccos_tuple = movie_reccos_tuples[0]
            index_images_tuple = movie_reccos_tuples[1]

            # Render page
            return render_template('index.html', selected_movie_image=selected_movie_image, selected_movie=selected_movie,
            movies=movie_titles_list, reccos=reccos_tuple[1:6], hiddenImages=index_images_tuple, display_error=0)
        else:
            return render_template('index.html', movies=movie_titles_list, display_error=0)

    except ValueError:
        display_error = 1
        return render_template('index.html', movies=movie_titles_list, display_error=1)

# Run Flask
if __name__=='__main__':
    app.run(debug=True)
        