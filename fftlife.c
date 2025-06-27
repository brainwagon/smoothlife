#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <unistd.h>
#include <SDL2/SDL.h>
#include <fftw3.h>

#define SIZE	(512)

/*======================================================================*/

const float ra = 10.0f;         /* outer radius */
const float rr = 3.0f;          /* ratio of radii */
const float rb = 10.f/3.0f; 	/* inner radius */
const float b1 = 0.257f;        /* birth1 */
const float b2 = 0.336f;        /* birth2 */
const float s1 = 0.365f;        /* survival1 */
const float s2 = 0.549f;        /* survival2 */
const float alpha_n = 0.028f;   /* outer sigmoid */
const float alpha_m = 0.147f;   /* inner sigmoid */
const float dt = 0.1f;          /* timestep */
const float scl = 1.0f/(SIZE*SIZE) ;

int
inside(float x0, float y0, float xc, float yc, float r)
{
    float l = (x0-xc) * (x0-xc) + (y0-yc) * (y0-yc) ;
    return l < (r * r) ;
}

float 
clamp(float x, float lo, float hi)
{
    if (x < lo) return lo ;
    if (x > hi) return hi ;
    return x ;
}

float 
sigma1(float x, float a, float alpha) 
{ 
    return 1.0f / ( 1.0f + expf( -(x-a)*4.0f/alpha ) );
}
        
float 
sigma2(float x, float a, float b, float alpha)
{
    return sigma1(x,a,alpha) * ( 1.0f-sigma1(x,b,alpha) );
}
        
float 
sigma_m(float x, float y, float m, float alpha)
{
    return x * ( 1.0f-sigma1(m,0.5f,alpha) ) + y * sigma1(m,0.5f,alpha); 
}
        
float 
s(float n, float m)
{
    return sigma_m( sigma2(n,b1,b2,alpha_n), 
	            sigma2(n,s1,s2,alpha_n), m, alpha_m);
}
        
float 
ramp_step(float x, float a, float ea)
{
    return clamp((x-a)/ea + 0.5f,0.0f,1.0f);
}


/*======================================================================*/

fftw_complex *ifilter ;		/* integrate the inner disk... */
fftw_complex *ofilter ;		/* integrate the outer annulus */
fftw_complex *state ;
fftw_complex *tmp ;
fftw_complex *isum ;
fftw_complex *osum ;

#ifdef __cplusplus
extern "C"
#endif
int main(int argc, char *argv[])
{
    fftw_plan fwd, irev, orev, iflt, oflt ;
    fftw_complex *ip, *op, *jp ;
    int x, y, g ;
    float area_in = 0.f ;
    float area_out = 0.f ;
    int running = 1 ;

    srand48(51064) ;

    fftw_init_threads() ;
    fftw_plan_with_nthreads(8) ;

    ifilter = (fftw_complex *) fftw_malloc(SIZE * SIZE * sizeof(fftw_complex)) ;
    ofilter = (fftw_complex *) fftw_malloc(SIZE * SIZE * sizeof(fftw_complex)) ;
    state = (fftw_complex *) fftw_malloc(SIZE * SIZE * sizeof(fftw_complex)) ;
    tmp = (fftw_complex *) fftw_malloc(SIZE * SIZE * sizeof(fftw_complex)) ;
    isum = (fftw_complex *) fftw_malloc(SIZE * SIZE * sizeof(fftw_complex)) ;
    osum = (fftw_complex *) fftw_malloc(SIZE * SIZE * sizeof(fftw_complex)) ;

    fftw_import_wisdom_from_filename("fftlife.wis") ;

    fprintf(stderr, "iflt...") ;
    iflt = fftw_plan_dft_2d(SIZE, SIZE, 
		ifilter, ifilter, FFTW_FORWARD, FFTW_ESTIMATE) ;
    fprintf(stderr, "oflt...") ;
    oflt = fftw_plan_dft_2d(SIZE, SIZE, 
		ofilter, ofilter, FFTW_FORWARD, FFTW_ESTIMATE) ;
    fprintf(stderr, "fwd...") ;
    fwd = fftw_plan_dft_2d(SIZE, SIZE, 
		state, tmp, FFTW_FORWARD, FFTW_ESTIMATE) ;
    fprintf(stderr, "irev...") ;
    irev = fftw_plan_dft_2d(SIZE, SIZE, 
		tmp, isum, FFTW_BACKWARD, FFTW_ESTIMATE) ;
    fprintf(stderr, "prev...") ;
    orev = fftw_plan_dft_2d(SIZE, SIZE, 
		tmp, osum, FFTW_BACKWARD, FFTW_ESTIMATE) ;
    fprintf(stderr, "done.\n") ;

    fftw_export_wisdom_to_filename("fftlife.wis") ;

#define UNIFORM
#ifdef  UNIFORM
    for (y=0, ip=state; y<SIZE; y++) {
	for (x=0; x<SIZE; x++, ip++) {
            if (x > SIZE/4 && x < 3 * SIZE/4 &&
                    y > SIZE / 4 && y < 3 * SIZE / 4) {
                *ip = drand48() ;
            } else {
                *ip = 0.0 ;
            }
        }
    }
#else
    /* initialize the state to random values... */
    for (y=0, ip=state; y<SIZE; y++) {
	for (x=0; x<SIZE; x++, ip++) {
	    int dx = x-SIZE/2 ;
	    int dy = y-SIZE/2 ;
	    if (dx * dx + dy * dy < 1450)
		*ip = 0.3*drand48() ;
	    else {
		dx = x - SIZE/4 ;
		dy = y - SIZE/4 ;
		if (dx * dx + dy * dy < 1450)
		    *ip = 0.3*drand48() ;
		else {
		    dx = x - 3*SIZE/4 ;
		    dy = y - SIZE/4 ;
		    if (dx * dx + dy * dy < 1450)
			*ip = 0.3*drand48() ;
		    else 
			*ip = drand48() ;
		}
	    }
	}
    }
#endif

#if 0
    for (y=0, ip=state; y<SIZE; y++) {
	for (x=0; x<SIZE; x++, ip++) {
	    float fx = (float) x / SIZE ;
	    float fy = (float) y / SIZE ;
	    
	    if (inside(fx, fy, 0.4, 0.3, 0.1) ||
		inside(fx, fy, 0.7, 0.6, 0.1) ||
		inside(fx, fy, 0.3, 0.8, 0.1)) {
		*ip = drand48() * 0.3 ;
	    } else {
		*ip = drand48() ;
	    }
	}
    }
#endif

    FILE *fp1 = fopen("ifilter.txt", "w") ;
    FILE *fp2 = fopen("ofilter.txt", "w") ;

    /* initialize and compute the filters */
    for (y=0, ip=ifilter, op=ofilter; y<SIZE; y++) {
	for (x=0; x<SIZE; x++, ip++, op++) {
	    float dx = x ;
	    float dy = y ;
	    if (dx > SIZE/2) dx -= SIZE ;
	    if (dy > SIZE/2) dy -= SIZE ;
	    float r = sqrt(dx*dx+dy*dy) ;
	    *ip = ramp_step(-r, -rb, 1.0f);
	    area_in += creal(*ip) ;
	    fprintf(fp1, "%f\n", creal(*ip)) ;

	    *op = ramp_step(-r, -ra, 1.0f) * ramp_step(r, rb, 1.0f);
	    area_out += creal(*op) ;
	    fprintf(fp2, "%f\n", creal(*op)) ;
	}
	fprintf(fp1, "\n") ;
	fprintf(fp2, "\n") ;
    }
    fclose(fp1) ;
    fclose(fp2) ;

#define IDX(x, y) (((x + SIZE) % SIZE) + ((y+SIZE) % SIZE) * SIZE)

    fftw_execute(iflt) ;
    fftw_execute(oflt) ;

    /* Now, use SDL  */

    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *window = SDL_CreateWindow("SmoothLife", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SIZE, SIZE, 0);
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB888, SDL_TEXTUREACCESS_STREAMING, SIZE, SIZE);

    /* allocate an RGB buffer */
    uint32_t *pixels = malloc(SIZE * SIZE * sizeof(uint32_t));

    for (g = 0; ; g++) {
	int t ;
	fprintf(stderr, "generation %03d\r", g) ;
	for (t=0; t<8; t++) {
	/* compute the fourier transform */
	fftw_execute(fwd) ;
	/* compute the inner integral */
	for (y=0, ip=tmp, jp=ifilter; y<SIZE; y++) {
	    for (x=0; x<SIZE; x++, ip++, jp++) {
		*ip *= *jp ;
	    }
	}
	fftw_execute(irev) ;

	/* compute the fourier transform */
	fftw_execute(fwd) ;
	/* compute the outer integral */
	for (y=0, ip=tmp, jp=ofilter; y<SIZE; y++) {
	    for (x=0; x<SIZE; x++, ip++, jp++) {
		*ip *= *jp ;
	    }
	}
	fftw_execute(orev) ;
	
#if 1
	ip = isum ;
	op = osum ;
	jp = state ;
	for (y=0; y<SIZE; y++) {
	    for (x=0; x<SIZE; x++) {
		float ia = creal(*ip) * scl / area_in ;
		float oa = creal(*op) * scl / area_out ;

		*jp = clamp(creal(*jp) + dt * (2.0f * s(oa, ia) - 1.0f), 
			    0.0f, 1.0f);
		ip++ ; op++ ;
		jp++ ;
	    }
	}

#endif
	}

#if 0
	/* that's it!  dump the frame! */

	char fname[80] ;
	sprintf(fname, "cjpeg -quality 90 > frames/frame.%04d.jpg", g) ;
	FILE *fp = popen(fname, "w") ;
	fprintf(fp, "P5\n%d %d\n%d\n", SIZE, SIZE, 255) ;

	for (y=0, ip=state; y<SIZE; y++) {
	    for (x=0; x<SIZE; x++, ip++) {
		int s = ((int) creal(*ip)) ;
		fputc(255*s, fp) ;
	    }
	}

	pclose(fp) ;
#else
	ip = state ;
	for (int y = 0; y < SIZE; ++y) {
	    for (int x = 0; x < SIZE; ++x) {
		float val = *ip++ ;
		uint8_t v = (uint8_t)(val * 255);
		pixels[y * SIZE + x] = (v << 16) | (v << 8) | v; // RGB
	    }
	}
	SDL_UpdateTexture(texture, NULL, pixels, SIZE * sizeof(uint32_t));

	SDL_RenderClear(renderer) ;
	SDL_RenderCopy(renderer, texture, NULL, NULL) ;
	SDL_RenderPresent(renderer) ;

	SDL_Event e;
	while (SDL_PollEvent(&e)) {
	    if (e.type == SDL_QUIT) running = 0;
	}

	if (! running) 
	    break ;
#endif
    }

    fftw_cleanup_threads() ;

    /* Destroy SDL */
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0 ;
}
