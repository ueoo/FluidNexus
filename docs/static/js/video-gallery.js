const VideoGallery = (function () {
  function handleVideoSwitch(galleryId, index) {
    const container = document.getElementById(galleryId);
    const items = container.querySelectorAll(".video-gallery-item");
    const buttons = container.parentElement.querySelector(".video-gallery-buttons").querySelectorAll(".button");
    const descriptions = container.querySelectorAll(".video-description");

    // Update items
    items.forEach((item, i) => {
      const video = item.querySelector("video");
      if (i === index) {
        item.classList.add("active");

        // LAZY-LOADING: If this <video> has a <source data-lazy="...">, set it now
        const lazySources = video.querySelectorAll("source[data-lazy]");
        lazySources.forEach((src) => {
          src.setAttribute("src", src.dataset.lazy);
          src.removeAttribute("data-lazy");
        });

        // Now load & play
        video.load();
        video.muted = true;
        video.currentTime = 0;
        video.play().catch((error) => {
          console.log("Autoplay was prevented:", error);
        });
      } else {
        item.classList.remove("active");
        // Pause & reset
        video.pause();
        video.currentTime = 0;
      }
    });

    // Update buttons
    buttons.forEach((button) => button.classList.remove("is-info"));
    buttons[index].classList.add("is-info");

    // Update descriptions (if any)
    descriptions.forEach((desc, i) => {
      if (i === index) {
        desc.classList.add("active");
      } else {
        desc.classList.remove("active");
      }
    });
  }

  function initGallery(galleryButtonsId, galleryCarouselId) {
    // Remove the code that pre-loads every video up front.
    // We only load on demand in handleVideoSwitch().

    // Set up button click handlers
    const buttonContainer = document.getElementById(galleryButtonsId);
    buttonContainer.querySelectorAll(".button").forEach((button) => {
      button.addEventListener("click", () => {
        const index = parseInt(button.dataset.videoIndex);
        handleVideoSwitch(galleryCarouselId, index);
      });
    });

    // Force an initial load for the “active” item if needed
    // e.g. if your first .video-gallery-item has "active" by default,
    // we can lazy-load it now:
    const activeItem = document.getElementById(galleryCarouselId).querySelector(".video-gallery-item.active");
    if (activeItem) {
      // figure out its index
      const allItems = activeItem.parentElement.querySelectorAll(".video-gallery-item");
      const activeIndex = [...allItems].indexOf(activeItem);
      handleVideoSwitch(galleryCarouselId, activeIndex >= 0 ? activeIndex : 0);
    }
  }

  return {
    init: function () {
      initGallery("gallery-buttons-nvs", "gallery-carousel-nvs");
      initGallery("gallery-buttons-fut", "gallery-carousel-fut");
      initGallery("gallery-buttons-resim", "gallery-carousel-resim");
      initGallery("gallery-buttons-resim-wild", "gallery-carousel-resim-wild");
      initGallery("gallery-buttons-interact", "gallery-carousel-interact");
      initGallery("gallery-buttons-abl-nvs", "gallery-carousel-abl-nvs");
      initGallery("gallery-buttons-abl-resim", "gallery-carousel-abl-resim");
      initGallery("gallery-buttons-dataset", "gallery-carousel-dataset");
    },
  };
})();

document.addEventListener("DOMContentLoaded", function () {
  VideoGallery.init();
});
