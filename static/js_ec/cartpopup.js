        function openSelectForm(styleNo) {

            document.getElementById("styleNo").value = styleNo;

            $("#selectModal").modal("show");

        }

        document.getElementById("selectForm").addEventListener("submit", function (e) {

            e.preventDefault();

            const data = {
                style_no: document.getElementById("styleNo").value,
                customer_name: document.getElementById("customerName").value,
                phone: document.getElementById("phone").value
            };

            console.log(data);

            // Send to Flask
            fetch("/save-selection", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(data)
            })
                .then(r => r.json())
                .then(res => {

                    alert(res.message);

                    $("#selectModal").modal("hide");

                    document.getElementById("selectForm").reset();

                });

        });
