const gulp = require('gulp');
const sass = require('gulp-sass')(require('sass'));
const pug = require('gulp-pug');
const browserSync = require('browser-sync').create();

// Compile SCSS to CSS
function compileSass(done) {
    gulp.src('src/scss/**/*.scss')
        .pipe(sass().on('error', sass.logError))
        .pipe(gulp.dest('static/css'))
        .pipe(browserSync.stream());
    done();
}

// Compile Pug to HTML
function compilePug(done) {
    gulp.src(['src/pug/**/*.pug'])
        .pipe(pug({ pretty: false }))
        .pipe(gulp.dest('static'))
        .pipe(browserSync.stream());
    done();
}

function serve(done) {
    browserSync.init({
        server: {
            baseDir: 'static',
        },
        port: 3000,
    });

    gulp.watch('src/scss/**/*.scss', compileSass);
    gulp.watch(['src/pug/**/*.pug'], compilePug);
    gulp.watch('static/*.html').on('change', browserSync.reload);
    done();
}

gulp.task('default', gulp.series(compileSass, compilePug, serve));