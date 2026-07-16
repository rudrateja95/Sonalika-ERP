        let page = 1;
        let loading = false;
        let finished = false;

        let preloadedProducts = [];

        // First load
        loadProducts();

        async function fetchPage(pageNo) {

            const response = await fetch(`/api/styles?brand=TASO&category=ladies%20ring&page=${pageNo}`);
            return await response.json();

        }

        function renderProducts(products) {

            let html = "";

            products.forEach(function (product) {

                html += `
    
    <div class="col-lg-3 col-md-6 col-sm-12 mb-4" >
    
        <div class="card h-100 shadow-sm border" >
    
            <!-- Image -->
<div class="image-box">

    <img
        src="data:image/jpeg;base64,${product.image}"
        alt="${product.style_no}"
        class="img-fluid product-img"
        style="height:260px; object-fit:contain;">

    <div class="image-overlay">

        <button
            class="btn btn-light rounded-circle"
            onclick="showImage('data:image/jpeg;base64,${product.image}')">

            <i class="fas fa-eye"></i>

        </button>

    </div>

</div>
    
            <!-- Card Body -->
            <div class="card-body d-flex flex-column">
    
                <h5 class="text-center font-weight-bold mb-1">
                    ${product.style_no}
                </h5>
    
    
                <table class="table table-bordered table-sm text-center mb-4">
    
                    <tbody>
    
                        <tr>
                            <th>Gross wt</th>
                            <td>${product.gold_wt != null ? Number(product.gold_wt).toFixed(3) : "-"}</td>
                        </tr>
    
                        <tr>
                            <th>Gold Net wt</th>
                            <td>${product.net_wt != null ? product.net_wt : "-"}</td>
                        </tr>
    
                        <tr>
                            <th>Diamond wt</th>
                            <td>${product.dia_wt != null ? product.dia_wt : "-"}</td>
                        </tr>
    
                        <tr>
                            <th>Diamond Pieces</th>
                            <td>${product.dia_pc != null ? product.dia_pc : "-"}</td>
                        </tr>

                        <tr>
                        <th>Color stn. pc</th>
                        <td>${product.cstn_pc != null ? product.cstn_pc : "-"}</td>
                    </tr>

                    <tr>
                        <th>Color stn. wt</th>
                        <td>${product.cstn_wt != null ? product.cstn_wt : "-"}<td>
                    </tr>
    
                    </tbody>
    
                </table>
    
                <!-- Button always at bottom -->
                <div class="mt-auto">
    
                     <button
                         class="btn btn-primary btn-block rounded-pill"
                         onclick="openSelectForm('${product.style_no}')">
                     
                         <i class="fas fa-shopping-cart mr-2"></i>
                         Select
                     
                     </button>
    
                </div>
    
            </div>
    
        </div>
    
    </div>
    
            `;

            });

            document.getElementById("products").insertAdjacentHTML("beforeend", html);

        }

        async function loadProducts() {

            if (loading || finished) return;

            loading = true;

            document.getElementById("loader").style.display = "block";

            // Use preloaded data if available
            if (preloadedProducts.length > 0) {

                renderProducts(preloadedProducts);

                preloadedProducts = [];

            } else {

                const data = await fetchPage(page);

                if (data.products.length == 0) {

                    finished = true;
                    document.getElementById("loader").innerHTML = "No More Products";
                    return;

                }

                renderProducts(data.products);

            }

            page++;

            loading = false;

            document.getElementById("loader").style.display = "none";

            // Preload next page in background
            fetchPage(page).then(function (data) {

                preloadedProducts = data.products;

            });

        }

        window.addEventListener("scroll", function () {

            if (window.innerHeight + window.pageYOffset >= document.body.offsetHeight - 300) {

                loadProducts();

            }

        });

