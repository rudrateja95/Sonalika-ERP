          function addOrder() {

            const container = document.getElementById("orderContainer");
            const clone = container.firstElementChild.cloneNode(true);

            // Reset selects
            clone.querySelectorAll("select").forEach(select => {
                select.selectedIndex = 0;
            });

            // Reset textarea
            clone.querySelectorAll("textarea").forEach(textarea => {
                textarea.value = "";
            });

            // Show remove button
            clone.querySelector(".remove-order").style.display = "block";

            container.appendChild(clone);
        }

        function removeOrder(btn) {
            btn.closest(".order-item").remove();
        }
