import numpy as np
import matplotlib.pyplot as plt

def compressed(image_name,sensibilty):
    # Load image and convert to grayscale
    img = plt.imread(image_name)
    gray = np.mean(img, axis=2)

    # Compute Fourier transform and shift the zero frequency to the center
    f = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f)

    # Compute the magnitude spectrum
    magnitude_spectrum = 20*np.log(np.abs(f_shift))

    # Take only the dominant frequencies
    magnitude_spectrum[magnitude_spectrum < np.percentile(magnitude_spectrum, sensibilty)] = 0

    # Filter out the irrelevant frequencies
    f_shift = f_shift * magnitude_spectrum

    # Shift the zero frequency back to the top-left corner
    f_shift_back = np.fft.ifftshift(f_shift)

    # Compute the inverse Fourier transform to get the filtered image
    filtered = np.real(np.fft.ifft2(f_shift_back))

    # Display the original and filtered images
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(gray, cmap='gray')
    ax[0].set_title('Original')
    ax[1].imshow(magnitude_spectrum, cmap='gray')
    ax[1].set_title('Magnitude Spectrum')
    ax[2].imshow(filtered, cmap='gray')
    ax[2].set_title('Filtered')
    plt.show()




compressed("Bewerb.png",99)


