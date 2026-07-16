        function showImage(imageSrc) {

            document.getElementById("popupImage").src = imageSrc;

            const modal = new bootstrap.Modal(
                document.getElementById("imageModal")
            );

            modal.show();

        }